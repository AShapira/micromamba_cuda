#!/usr/bin/env python3
"""
Aggregate OSM / non-OSM building counts per division polygon using DuckDB.

Each summary file is processed in parallel worker processes. Every worker keeps
its own DuckDB connection, streams Parquet row groups, filters candidate
divisions by bounding boxes, and evaluates ST_Contains inside SQL while logging
progress every million rows.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from datetime import datetime

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from shapely import from_wkb as shapely_from_wkb, to_wkb as shapely_to_wkb, union_all as shapely_union_all

LOGGER = logging.getLogger("divisions_summary_duckdb")
LOG_PROGRESS_EVERY = 1_000_000

WORKER_CONN: duckdb.DuckDBPyConnection | None = None
WORKER_DIVISION_COUNT = 0
WORKER_CHUNK_SIZE = 100_000
WORKER_HAS_UNION = False
WORKER_LOGGER = logging.getLogger("divisions_summary_duckdb")
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
FROM divisions
WHERE
    bbox_xmax >= $minx
    AND bbox_xmin <= $maxx
    AND bbox_ymax >= $miny
    AND bbox_ymin <= $maxy;
"""

UNION_INTERSECTS_QUERY = """
SELECT ST_Intersects(
    (SELECT geom FROM divisions_union LIMIT 1),
    ST_MakeEnvelope($minx, $miny, $maxx, $maxy)
);
"""

UNION_POINT_EXISTS_QUERY = """
SELECT 1
FROM chunk_points cp
CROSS JOIN divisions_union du
WHERE ST_Contains(du.geom, cp.geom)
LIMIT 1;
"""

AGG_QUERY = """
WITH chunk_xy AS (
    SELECT
        geom,
        is_osm,
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
filtered_divisions AS (
    SELECT *
    FROM divisions
    WHERE
        bbox_xmax >= (SELECT minx FROM bounds)
        AND bbox_xmin <= (SELECT maxx FROM bounds)
        AND bbox_ymax >= (SELECT miny FROM bounds)
        AND bbox_ymin <= (SELECT maxy FROM bounds)
),
cover_candidates AS (
    SELECT
        fd.division_idx
    FROM filtered_divisions fd, bounds b
    WHERE ST_Covers(
        fd.geom,
        ST_MakeEnvelope(b.minx, b.miny, b.maxx, b.maxy)
    )
),
cover_counts AS (
    SELECT
        cover.division_idx,
        SUM(CASE WHEN chunk_points.is_osm THEN 1 ELSE 0 END) AS osm_add,
        SUM(CASE WHEN chunk_points.is_osm THEN 0 ELSE 1 END) AS non_osm_add
    FROM chunk_points
    CROSS JOIN (SELECT division_idx FROM cover_candidates LIMIT 1) AS cover
    WHERE (SELECT COUNT(*) FROM cover_candidates) = 1
    GROUP BY cover.division_idx
),
counts AS (
    SELECT
        fd.division_idx,
        SUM(CASE WHEN chunk_xy.is_osm THEN 1 ELSE 0 END) AS osm_add,
        SUM(CASE WHEN chunk_xy.is_osm THEN 0 ELSE 1 END) AS non_osm_add
    FROM chunk_xy
    JOIN filtered_divisions fd
        ON chunk_xy.x BETWEEN fd.bbox_xmin AND fd.bbox_xmax
       AND chunk_xy.y BETWEEN fd.bbox_ymin AND fd.bbox_ymax
       AND ST_Contains(fd.geom, chunk_xy.geom)
    WHERE (SELECT COUNT(*) FROM cover_candidates) <> 1
    GROUP BY fd.division_idx
)
SELECT division_idx, osm_add, non_osm_add FROM cover_counts
UNION ALL
SELECT division_idx, osm_add, non_osm_add FROM counts;
"""


@dataclass(frozen=True)
class Task:
    input_path: Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "summary_dir",
        type=Path,
        help="Directory that contains the buildings summary GeoParquet files.",
    )
    parser.add_argument(
        "divisions_path",
        type=Path,
        help="GeoParquet file with division polygons (same schema as country_area).",
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
        "--max-files",
        type=int,
        default=None,
        help="Optional limit on the number of summary files to process (useful for testing).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Number of rows per chunk when slicing a Parquet row group (default: 100k).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args(argv)


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


def _has_bbox_column(parquet_path: Path) -> bool:
    schema = pq.read_schema(parquet_path)
    return "bbox" in schema.names


def _extract_bbox_from_struct(values: pd.Series, key: str) -> pd.Series:
    return values.apply(lambda b: b.get(key) if isinstance(b, dict) else np.nan)


def load_divisions(divisions_path: Path) -> tuple[pd.DataFrame, dict]:
    table = pq.read_table(divisions_path)
    pdf = table.to_pandas()
    if "geometry" not in pdf.columns:
        raise ValueError("Divisions parquet must contain a 'geometry' column (WKB).")

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

    pdf["division_idx"] = np.arange(len(pdf), dtype=np.int32)
    pdf["osm_count"] = np.int64(0)
    pdf["non_osm_count"] = np.int64(0)

    union_geom = shapely_from_wkb(pdf["geometry"].to_numpy())
    union = shapely_union_all(union_geom)

    payload = {
        "division_idx": pdf["division_idx"].to_numpy(dtype=np.int32),
        "geom_wkb": pdf["geometry"].to_numpy(),
        "bbox_xmin": pdf["_bbox_xmin"].to_numpy(dtype=float),
        "bbox_xmax": pdf["_bbox_xmax"].to_numpy(dtype=float),
        "bbox_ymin": pdf["_bbox_ymin"].to_numpy(dtype=float),
        "bbox_ymax": pdf["_bbox_ymax"].to_numpy(dtype=float),
        "union_wkb": shapely_to_wkb(union, hex=False) if union else None,
    }
    return pdf, payload


def _init_worker(payload: dict) -> None:
    configure_logging(payload["log_level"], Path(payload["log_path"]))
    global WORKER_CONN, WORKER_DIVISION_COUNT, WORKER_CHUNK_SIZE, WORKER_HAS_UNION

    con = duckdb.connect(database=":memory:")
    con.execute("INSTALL spatial;")
    con.execute("LOAD spatial;")
    threads = payload.get("threads_per_worker")
    if threads:
        con.execute(f"PRAGMA threads={max(1, threads)};")

    divisions_table = pa.table(
        {
            "division_idx": pa.array(payload["division_idx"], type=pa.int32()),
            "geom_wkb": pa.array(payload["geom_wkb"], type=pa.binary()),
            "bbox_xmin": pa.array(payload["bbox_xmin"], type=pa.float64()),
            "bbox_xmax": pa.array(payload["bbox_xmax"], type=pa.float64()),
            "bbox_ymin": pa.array(payload["bbox_ymin"], type=pa.float64()),
            "bbox_ymax": pa.array(payload["bbox_ymax"], type=pa.float64()),
        }
    )

    con.register("divisions_src", divisions_table)
    con.execute(
        """
        CREATE TEMP TABLE divisions AS
        SELECT
            division_idx,
            ST_GeomFromWKB(geom_wkb) AS geom,
            bbox_xmin,
            bbox_xmax,
            bbox_ymin,
            bbox_ymax
        FROM divisions_src;
        """
    )
    con.unregister("divisions_src")

    union_wkb = payload.get("union_wkb")
    if union_wkb:
        union_table = pa.table({"geom": pa.array([union_wkb], type=pa.binary())})
        con.register("divisions_union_src", union_table)
        con.execute(
            """
            CREATE TEMP TABLE divisions_union AS
            SELECT ST_GeomFromWKB(geom) AS geom FROM divisions_union_src;
            """
        )
        con.unregister("divisions_union_src")
        WORKER_HAS_UNION = True
    else:
        WORKER_HAS_UNION = False

    WORKER_CONN = con
    WORKER_DIVISION_COUNT = len(payload["division_idx"])
    WORKER_CHUNK_SIZE = max(1, int(payload.get("chunk_size", WORKER_CHUNK_SIZE)))


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
                COALESCE(is_osm, FALSE) AS is_osm
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


def _process_summary_file(path: Path) -> tuple[str, np.ndarray, np.ndarray]:
    logger = logging.getLogger("divisions_summary_duckdb")
    logger.info("Starting %s", path.name)

    parquet = pq.ParquetFile(path)
    processed = 0
    next_progress = LOG_PROGRESS_EVERY
    osm_counts = np.zeros(WORKER_DIVISION_COUNT, dtype=np.int64)
    non_osm_counts = np.zeros(WORKER_DIVISION_COUNT, dtype=np.int64)

    for batch in parquet.iter_batches(
        columns=["geometry", "is_osm"], batch_size=WORKER_CHUNK_SIZE
    ):
        table = pa.Table.from_batches([batch])
        result = _apply_chunk(table)
        if result is not None:
            division_idx = np.asarray(result.column(0).to_numpy(zero_copy_only=False), dtype=np.int64)
            osm_add = np.asarray(result.column(1).to_numpy(zero_copy_only=False), dtype=np.int64)
            non_add = np.asarray(result.column(2).to_numpy(zero_copy_only=False), dtype=np.int64)
            osm_counts[division_idx] += osm_add
            non_osm_counts[division_idx] += non_add
        processed += batch.num_rows
        while processed >= next_progress:
            logger.info("%s: processed %s rows", path.name, next_progress)
            next_progress += LOG_PROGRESS_EVERY

    logger.info("%s: finished %s rows", path.name, parquet.metadata.num_rows)
    return path.name, osm_counts, non_osm_counts


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    summary_dir = args.summary_dir.expanduser()
    divisions_path = args.divisions_path.expanduser()
    if not summary_dir.exists():
        raise FileNotFoundError(f"Summary directory {summary_dir} does not exist.")
    if not divisions_path.exists():
        raise FileNotFoundError(f"Divisions file {divisions_path} does not exist.")

    summary_files = sorted(path for path in summary_dir.glob("*.parquet") if path.is_file())
    if args.max_files is not None:
        summary_files = summary_files[: max(0, args.max_files)]
    if not summary_files:
        raise FileNotFoundError(f"No summary parquet files found in {summary_dir}.")

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_path = divisions_path.parent / f"{divisions_path.stem}_osm_summary_duckdb_{timestamp}.log"
    configure_logging(args.log_level, log_path)
    LOGGER.debug("Logging to %s", log_path)

    divisions_df, payload = load_divisions(divisions_path)
    tasks = [Task(path) for path in summary_files]
    worker_count = max(1, min(args.max_workers, len(tasks)))
    LOGGER.info("Processing %s file(s) with %s worker(s)", len(tasks), worker_count)

    os.makedirs(log_path.parent, exist_ok=True)
    payload.update(
        {
            "threads_per_worker": args.threads_per_worker,
            "log_path": str(log_path),
            "log_level": args.log_level,
            "chunk_size": args.chunk_size,
        }
    )

    osm_total = np.zeros(len(divisions_df), dtype=np.int64)
    non_osm_total = np.zeros(len(divisions_df), dtype=np.int64)

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_init_worker,
        initargs=(payload,),
    ) as executor:
        futures = {executor.submit(_process_summary_file, task.input_path): task for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            name, osm_counts, non_counts = future.result()
            LOGGER.info("Merged results for %s", name)
            osm_total += osm_counts
            non_osm_total += non_counts

    divisions_df["osm_count"] = osm_total
    divisions_df["non_osm_count"] = non_osm_total
    drop_cols = [col for col in divisions_df.columns if col.startswith("_bbox_") or col == "division_idx"]
    divisions_df = divisions_df.drop(columns=drop_cols)

    output_path = divisions_path.with_name(f"{divisions_path.stem}_osm_summary{divisions_path.suffix}")
    pq.write_table(pa.Table.from_pandas(divisions_df, preserve_index=False), output_path, compression="ZSTD")
    LOGGER.info("Wrote %s", output_path)
    LOGGER.info("Log written to %s", log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
