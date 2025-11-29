#!/usr/bin/env python3
"""
Aggregate building dataset counts per region using DuckDB with spatial filtering.

The script reads the lightweight building summary GeoParquet files produced by
`summarize_buildings_by_source.py` (geometry point + s1/s2 dataset ids) and
counts how many buildings from each dataset fall into every region polygon.
Only `s1` is used; `s2` is ignored. A column is written for every dataset in
DATASET_ID_MAP plus an `unknown` column for ids that are missing or not mapped.

Why it is efficient
- Streams Parquet row groups and slices them into configurable chunks to keep
  memory bounded while sustaining throughput.
- Keeps the regions table resident inside each DuckDB worker process, uses
  bbox pre-filters and ST_Contains in SQL to avoid Python loops.
- Optional union geometry lets workers quickly skip chunks whose bbox does not
  intersect any region.
- Parallel processing across summary files; DuckDB can also use multiple
  threads per worker when requested.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from shapely import from_wkb as shapely_from_wkb, to_wkb as shapely_to_wkb, union_all as shapely_union_all

LOGGER = logging.getLogger("region_dataset_counts_duckdb")
LOG_PROGRESS_EVERY = 1_000_000

# Dataset ID mapping (lowercased keys) from summarize_buildings_by_source.py
DATASET_ID_MAP: dict[str, int] = {
    "openstreetmap": 1,
    "esri community maps": 2,
    "instituto geográfico nacional (españa)": 3,
    "google open buildings": 5,
    "microsoft ml buildings": 6,
    "doi:10.5281/zenodo.8174931": 7,
    "usgs lidar": 10,
}
UNKNOWN_ID = 0


@dataclass(frozen=True)
class DatasetColumn:
    dataset_id: int
    column_name: str
    label: str


def _slugify(name: str) -> str:
    """ASCII-safe column name derived from a dataset label."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^0-9a-z]+", "_", ascii_name.lower()).strip("_")
    return slug or "dataset"


def _build_dataset_columns() -> tuple[list[DatasetColumn], dict[int, int]]:
    columns: list[DatasetColumn] = [DatasetColumn(dataset_id=UNKNOWN_ID, column_name="unknown", label="unknown")]
    for name, ds_id in sorted(DATASET_ID_MAP.items(), key=lambda item: item[1]):
        columns.append(DatasetColumn(dataset_id=ds_id, column_name=_slugify(name), label=name))
    idx_map = {col.dataset_id: idx for idx, col in enumerate(columns)}
    return columns, idx_map


def _has_bbox_column(parquet_path: Path) -> bool:
    schema = pq.read_schema(parquet_path)
    return "bbox" in schema.names


def _extract_bbox_from_struct(values: pd.Series, key: str) -> pd.Series:
    return values.apply(lambda b: b.get(key) if isinstance(b, dict) else np.nan)


def load_regions(regions_path: Path) -> tuple[pd.DataFrame, dict]:
    table = pq.read_table(regions_path)
    pdf = table.to_pandas()
    if "geometry" not in pdf.columns:
        raise ValueError("Regions parquet must contain a 'geometry' column (WKB).")

    if "bbox" in pdf.columns:
        pdf["_bbox_xmin"] = _extract_bbox_from_struct(pdf["bbox"], "xmin")
        pdf["_bbox_xmax"] = _extract_bbox_from_struct(pdf["bbox"], "xmax")
        pdf["_bbox_ymin"] = _extract_bbox_from_struct(pdf["bbox"], "ymin")
        pdf["_bbox_ymax"] = _extract_bbox_from_struct(pdf["bbox"], "ymax")
    elif {"xmin", "xmax", "ymin", "ymax"}.issubset(pdf.columns):
        pdf["_bbox_xmin"] = pd.to_numeric(pdf["xmin"], errors="coerce")
        pdf["_bbox_xmax"] = pd.to_numeric(pdf["xmax"], errors="coerce")
        pdf["_bbox_ymin"] = pd.to_numeric(pdf["ymin"], errors="coerce")
        pdf["_bbox_ymax"] = pd.to_numeric(pdf["ymax"], errors="coerce")
    else:
        geoms = shapely_from_wkb(pdf["geometry"].to_numpy())
        bounds = np.array([geom.bounds for geom in geoms])
        pdf["_bbox_xmin"] = bounds[:, 0]
        pdf["_bbox_ymin"] = bounds[:, 1]
        pdf["_bbox_xmax"] = bounds[:, 2]
        pdf["_bbox_ymax"] = bounds[:, 3]

    pdf["_bbox_xmin"] = pdf["_bbox_xmin"].fillna(0.0)
    pdf["_bbox_xmax"] = pdf["_bbox_xmax"].fillna(0.0)
    pdf["_bbox_ymin"] = pdf["_bbox_ymin"].fillna(0.0)
    pdf["_bbox_ymax"] = pdf["_bbox_ymax"].fillna(0.0)

    pdf["region_idx"] = np.arange(len(pdf), dtype=np.int32)

    union_geom = shapely_from_wkb(pdf["geometry"].to_numpy())
    union = shapely_union_all(union_geom)

    payload = {
        "region_idx": pdf["region_idx"].to_numpy(dtype=np.int32),
        "geom_wkb": pdf["geometry"].to_numpy(),
        "bbox_xmin": pdf["_bbox_xmin"].to_numpy(dtype=float),
        "bbox_xmax": pdf["_bbox_xmax"].to_numpy(dtype=float),
        "bbox_ymin": pdf["_bbox_ymin"].to_numpy(dtype=float),
        "bbox_ymax": pdf["_bbox_ymax"].to_numpy(dtype=float),
        "union_wkb": shapely_to_wkb(union, hex=False) if union else None,
    }
    return pdf, payload


@dataclass(frozen=True)
class Task:
    input_path: Path


WORKER_CONN: duckdb.DuckDBPyConnection | None = None
WORKER_REGION_COUNT = 0
WORKER_CHUNK_SIZE = 100_000
WORKER_HAS_UNION = False
WORKER_DATASET_IDX: dict[int, int] = {}
VALID_DATASET_IDS = sorted({UNKNOWN_ID, *DATASET_ID_MAP.values()})

BOUNDS_QUERY = """
SELECT
    MIN(ST_X(geom)) AS minx,
    MAX(ST_X(geom)) AS maxx,
    MIN(ST_Y(geom)) AS miny,
    MAX(ST_Y(geom)) AS maxy
FROM chunk_points;
"""

CANDIDATE_COUNT_QUERY = """
SELECT COUNT(*)
FROM regions
WHERE
    bbox_xmax >= $minx
    AND bbox_xmin <= $maxx
    AND bbox_ymax >= $miny
    AND bbox_ymin <= $maxy;
"""

UNION_INTERSECTS_QUERY = """
SELECT ST_Intersects(
    (SELECT geom FROM regions_union LIMIT 1),
    ST_MakeEnvelope($minx, $miny, $maxx, $maxy)
);
"""

UNION_POINT_EXISTS_QUERY = """
SELECT 1
FROM chunk_points cp
CROSS JOIN regions_union ru
WHERE ST_Contains(ru.geom, cp.geom)
LIMIT 1;
"""

VALID_IDS_SQL = ", ".join(str(i) for i in VALID_DATASET_IDS)

AGG_QUERY = f"""
WITH chunk_xy AS (
    SELECT
        geom,
        CASE
            WHEN dataset_id IN ({VALID_IDS_SQL}) THEN dataset_id
            ELSE {UNKNOWN_ID}
        END AS dataset_id,
        ST_X(geom) AS x,
        ST_Y(geom) AS y
    FROM chunk_points
),
bounds AS (
    SELECT
        MIN(x) AS minx,
        MAX(x) AS maxx,
        MIN(y) AS miny,
        MAX(y) AS maxy
    FROM chunk_xy
),
filtered_regions AS (
    SELECT *
    FROM regions
    WHERE
        bbox_xmax >= (SELECT minx FROM bounds)
        AND bbox_xmin <= (SELECT maxx FROM bounds)
        AND bbox_ymax >= (SELECT miny FROM bounds)
        AND bbox_ymin <= (SELECT maxy FROM bounds)
),
cover_candidates AS (
    SELECT
        fr.region_idx
    FROM filtered_regions fr, bounds b
    WHERE ST_Covers(
        fr.geom,
        ST_MakeEnvelope(b.minx, b.miny, b.maxx, b.maxy)
    )
),
cover_counts AS (
    SELECT
        cover.region_idx,
        chunk_xy.dataset_id,
        COUNT(*) AS dataset_add
    FROM chunk_xy
    CROSS JOIN (SELECT region_idx FROM cover_candidates LIMIT 1) AS cover
    WHERE (SELECT COUNT(*) FROM cover_candidates) = 1
    GROUP BY cover.region_idx, chunk_xy.dataset_id
),
counts AS (
    SELECT
        fr.region_idx,
        chunk_xy.dataset_id,
        COUNT(*) AS dataset_add
    FROM chunk_xy
    JOIN filtered_regions fr
        ON chunk_xy.x BETWEEN fr.bbox_xmin AND fr.bbox_xmax
       AND chunk_xy.y BETWEEN fr.bbox_ymin AND fr.bbox_ymax
       AND ST_Contains(fr.geom, chunk_xy.geom)
    WHERE (SELECT COUNT(*) FROM cover_candidates) <> 1
    GROUP BY fr.region_idx, chunk_xy.dataset_id
)
SELECT region_idx, dataset_id, dataset_add FROM cover_counts
UNION ALL
SELECT region_idx, dataset_id, dataset_add FROM counts;
"""


def configure_logging(level: str, log_file: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def _init_worker(payload: dict) -> None:
    configure_logging(payload["log_level"], Path(payload["log_path"]))
    global WORKER_CONN, WORKER_REGION_COUNT, WORKER_CHUNK_SIZE, WORKER_HAS_UNION, WORKER_DATASET_IDX

    con = duckdb.connect(database=":memory:")
    con.execute("INSTALL spatial;")
    con.execute("LOAD spatial;")
    threads = payload.get("threads_per_worker")
    if threads:
        con.execute(f"PRAGMA threads={max(1, threads)};")

    regions_table = pa.table(
        {
            "region_idx": pa.array(payload["region_idx"], type=pa.int32()),
            "geom_wkb": pa.array(payload["geom_wkb"], type=pa.binary()),
            "bbox_xmin": pa.array(payload["bbox_xmin"], type=pa.float64()),
            "bbox_xmax": pa.array(payload["bbox_xmax"], type=pa.float64()),
            "bbox_ymin": pa.array(payload["bbox_ymin"], type=pa.float64()),
            "bbox_ymax": pa.array(payload["bbox_ymax"], type=pa.float64()),
        }
    )

    con.register("regions_src", regions_table)
    con.execute(
        """
        CREATE TEMP TABLE regions AS
        SELECT
            region_idx,
            ST_GeomFromWKB(geom_wkb) AS geom,
            bbox_xmin,
            bbox_xmax,
            bbox_ymin,
            bbox_ymax
        FROM regions_src;
        """
    )
    con.unregister("regions_src")

    union_wkb = payload.get("union_wkb")
    if union_wkb:
        union_table = pa.table({"geom": pa.array([union_wkb], type=pa.binary())})
        con.register("regions_union_src", union_table)
        con.execute(
            """
            CREATE TEMP TABLE regions_union AS
            SELECT ST_GeomFromWKB(geom) AS geom FROM regions_union_src;
            """
        )
        con.unregister("regions_union_src")
        WORKER_HAS_UNION = True
    else:
        WORKER_HAS_UNION = False

    WORKER_CONN = con
    WORKER_REGION_COUNT = len(payload["region_idx"])
    WORKER_CHUNK_SIZE = max(1, int(payload.get("chunk_size", WORKER_CHUNK_SIZE)))
    WORKER_DATASET_IDX = payload["dataset_idx_map"]


def _apply_chunk(chunk: pa.Table) -> pa.Table | None:
    if chunk.num_rows == 0 or WORKER_CONN is None:
        return None

    WORKER_CONN.register("summary_chunk", chunk)
    try:
        WORKER_CONN.execute(
            """
            CREATE TEMP TABLE chunk_points AS
            SELECT
                ST_GeomFromWKB(geometry) AS geom,
                COALESCE(s1, 0)::INT AS dataset_id
            FROM summary_chunk
            WHERE geometry IS NOT NULL;
            """
        )

        bounds = WORKER_CONN.execute(BOUNDS_QUERY).fetchone()
        if bounds is None or any(value is None for value in bounds):
            return None
        minx, maxx, miny, maxy = bounds

        if WORKER_HAS_UNION:
            intersects = WORKER_CONN.execute(
                UNION_INTERSECTS_QUERY,
                {"minx": minx, "maxx": maxx, "miny": miny, "maxy": maxy},
            ).fetchone()[0]
            if not intersects:
                return None
            hit = WORKER_CONN.execute(UNION_POINT_EXISTS_QUERY).fetchone()
            if hit is None:
                return None

        candidate_count = WORKER_CONN.execute(
            CANDIDATE_COUNT_QUERY,
            {"minx": minx, "maxx": maxx, "miny": miny, "maxy": maxy},
        ).fetchone()[0]
        if candidate_count == 0:
            return None

        result = WORKER_CONN.execute(AGG_QUERY).fetch_arrow_table()
        return result if result.num_rows else None
    finally:
        WORKER_CONN.execute("DROP TABLE IF EXISTS chunk_points")
        WORKER_CONN.unregister("summary_chunk")


def _process_summary_file(path: Path) -> tuple[str, np.ndarray]:
    logger = logging.getLogger("region_dataset_counts_duckdb")
    logger.info("Starting %s", path.name)

    parquet = pq.ParquetFile(path)
    processed = 0
    next_progress = LOG_PROGRESS_EVERY
    counts = np.zeros((WORKER_REGION_COUNT, len(WORKER_DATASET_IDX)), dtype=np.int64)

    for batch in parquet.iter_batches(columns=["geometry", "s1"], batch_size=WORKER_CHUNK_SIZE):
        table = pa.Table.from_batches([batch])
        result = _apply_chunk(table)
        if result is not None:
            region_idx = np.asarray(result.column(0).to_numpy(zero_copy_only=False), dtype=np.int64)
            dataset_ids = np.asarray(result.column(1).to_numpy(zero_copy_only=False), dtype=np.int64)
            dataset_idx = np.fromiter(
                (WORKER_DATASET_IDX[int(ds_id)] for ds_id in dataset_ids),
                dtype=np.int64,
                count=len(dataset_ids),
            )
            add_counts = np.asarray(result.column(2).to_numpy(zero_copy_only=False), dtype=np.int64)
            np.add.at(counts, (region_idx, dataset_idx), add_counts)
        processed += batch.num_rows
        while processed >= next_progress:
            logger.info("%s: processed %s rows", path.name, next_progress)
            next_progress += LOG_PROGRESS_EVERY

    logger.info("%s: finished %s rows", path.name, parquet.metadata.num_rows)
    return path.name, counts


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "summary_dir",
        type=Path,
        help="Directory containing building summary GeoParquet files (from summarize_buildings_by_source.py).",
    )
    parser.add_argument(
        "regions_path",
        type=Path,
        help="GeoParquet file with region polygons (same schema as region_area).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path (default: regions_path with _dataset_counts suffix).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="Number of parallel worker processes (default: half the CPUs).",
    )
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=None,
        help="Optional DuckDB thread count inside each worker (default: DuckDB automatic).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Number of rows per chunk when slicing a Parquet row group (default: 100k).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit on the number of summary files to process (useful for testing).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    summary_dir = args.summary_dir.expanduser()
    regions_path = args.regions_path.expanduser()
    if not summary_dir.exists():
        raise FileNotFoundError(f"Summary directory {summary_dir} does not exist.")
    if not regions_path.exists():
        raise FileNotFoundError(f"Regions file {regions_path} does not exist.")

    summary_files = sorted(path for path in summary_dir.glob("*.parquet") if path.is_file())
    if args.max_files is not None:
        summary_files = summary_files[: max(0, args.max_files)]
    if not summary_files:
        raise FileNotFoundError(f"No summary parquet files found in {summary_dir}.")

    output_path = (
        args.output.expanduser()
        if args.output
        else regions_path.with_name(f"{regions_path.stem}_dataset_counts{regions_path.suffix}")
    )
    timestamp = output_path.stem + ".log"
    log_path = output_path.with_name(timestamp)
    configure_logging(args.log_level, log_path)
    LOGGER.debug("Logging to %s", log_path)

    regions_df, payload = load_regions(regions_path)
    dataset_columns, dataset_idx_map = _build_dataset_columns()
    payload.update(
        {
            "threads_per_worker": args.threads_per_worker,
            "log_path": str(log_path),
            "log_level": args.log_level,
            "chunk_size": args.chunk_size,
            "dataset_idx_map": dataset_idx_map,
        }
    )

    tasks = [Task(path) for path in summary_files]
    worker_count = max(1, min(args.max_workers, len(tasks)))
    LOGGER.info("Processing %s file(s) with %s worker(s)", len(tasks), worker_count)

    total_counts = np.zeros((len(regions_df), len(dataset_columns)), dtype=np.int64)

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_init_worker,
        initargs=(payload,),
    ) as executor:
        futures = {executor.submit(_process_summary_file, task.input_path): task for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            name, counts = future.result()
            LOGGER.info("Merged results for %s", name)
            total_counts += counts

    for idx, col in enumerate(dataset_columns):
        regions_df[col.column_name] = total_counts[:, idx]

    drop_cols = [col for col in regions_df.columns if col.startswith("_bbox_") or col == "region_idx"]
    regions_df = regions_df.drop(columns=drop_cols)

    pq.write_table(pa.Table.from_pandas(regions_df, preserve_index=False), output_path, compression="ZSTD")
    LOGGER.info("Wrote %s", output_path)
    LOGGER.info("Log written to %s", log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
