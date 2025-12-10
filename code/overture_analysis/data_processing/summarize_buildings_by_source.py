#!/usr/bin/env python3
"""
Summarize Overture buildings GeoParquet files into lightweight point layers with source IDs.

For every GeoParquet in the input directory a summary GeoParquet is written to
data/results/buildings_source_summary (or a user supplied directory). Each summary
row corresponds to one building row and contains:
* geometry: Point geometry representing the center of the original bbox
* s1: ID of the first source dataset (0 for unlisted/unknown)
* s2: ID of the second source dataset when present (0 when not present)
* area_sqm: Geodesic area of the building footprint in square meters (stored as int)
* vertex_count: Count of coordinate vertices in the footprint (excluding closing coordinate duplicates)
* height: Height value from the source data (if present)

The script also writes a JSON log with aggregate and per-file dataset_counts and
combination_counts, and records execution time.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from pyproj import Geod
import pyarrow as pa
import pyarrow.parquet as pq
from shapely import from_wkb as shapely_from_wkb
from shapely import get_num_coordinates
from shapely import points as shapely_points
from shapely import to_wkb as shapely_to_wkb

LOGGER = logging.getLogger("buildings_source_summary")

DEFAULT_RESULTS_DIR = Path("data/results/buildings_source_summary")
LOG_FILENAME = "buildings_source_summary.log"
OUTPUT_BASENAME_RE = re.compile(r"^(part-\d+)")
DEFAULT_BATCH_ROWS = 200_000
LOG_PROGRESS_EVERY = 1_000_000

# Dataset ID mapping (lowercased keys) based on Conflation Priority from docks
DATASET_ID_MAP: dict[str, int] = {
    "openstreetmap": 1,
    "esri community maps": 2,
    "instituto geográfico nacional (españa)": 3,
    "google open buildings": 5,
    "microsoft ml buildings": 6,
    "doi:10.5281/zenodo.8174931": 7,
    "usgs lidar": 10,
}

ID_TO_NAME = {v: k for k, v in DATASET_ID_MAP.items()}
UNKNOWN_ID = 0
GEOD = Geod(ellps="WGS84")


def _build_geo_metadata(crs: str = "EPSG:4326") -> dict[bytes, bytes]:
    geo = {
        "version": "1.1.0",
        "primary_column": "geometry",
        "crs": {
            "type": "name",
            "properties": {"name": crs},
        },
        "columns": {
            "geometry": {
                "encoding": "WKB",
                "geometry_types": ["Point"],
                "crs": crs,
            }
        },
    }
    return {b"geo": json.dumps(geo).encode("utf-8")}


def _derive_output_path(input_path: Path, results_dir: Path) -> Path:
    match = OUTPUT_BASENAME_RE.match(input_path.name)
    prefix = match.group(1) if match else input_path.stem
    return results_dir / f"{prefix}-summary.parquet"


@dataclass(frozen=True)
class Task:
    input_path: Path
    output_path: Path
    skip_existing: bool
    batch_rows: int


@dataclass
class FileCounts:
    dataset_counts: Counter[str]
    combination_counts: Counter[str]
    rows: int = 0


def _make_summary_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("geometry", pa.binary()),
            pa.field("s1", pa.int8()),
            pa.field("s2", pa.int8()),
            pa.field("area_sqm", pa.int32()),
            pa.field("vertex_count", pa.int16()),
            pa.field("height", pa.int16()),
        ]
    ).with_metadata(_build_geo_metadata())


def _normalize_sources(value: object) -> list[str]:
    """Return normalized dataset names (lowercased) in input order."""
    if value is None:
        return []
    entries: Iterable[object] = value if isinstance(value, list) else (value,)
    names: list[str] = []
    for entry in entries:
        dataset: str | None = None
        if isinstance(entry, dict):
            raw = entry.get("dataset") or entry.get("dataset_name")
            dataset = str(raw).strip() if raw is not None else None
        elif isinstance(entry, str):
            dataset = entry.strip()
        if dataset:
            names.append(dataset.lower())
    return names


def _source_ids(values: object) -> tuple[int, int, list[int]]:
    names = _normalize_sources(values)
    ids: list[int] = []
    for name in names:
        ids.append(DATASET_ID_MAP.get(name, UNKNOWN_ID))
    first = ids[0] if ids else UNKNOWN_ID
    second = ids[1] if len(ids) > 1 else UNKNOWN_ID
    unique_sorted = sorted(set(ids))
    return first, second, unique_sorted


def _combo_label(id_list: list[int]) -> str:
    if not id_list:
        return "none"
    return "+".join(ID_TO_NAME.get(i, "other") for i in id_list)


def _unique_vertex_count(geom) -> int | None:
    """Return vertex count excluding duplicated closing coordinate for polygon rings."""
    if geom is None:
        return None
    if geom.geom_type == "Polygon":
        count = max(len(geom.exterior.coords) - 1, 0)
        for ring in geom.interiors:
            count += max(len(ring.coords) - 1, 0)
        return count
    if geom.geom_type == "MultiPolygon":
        total = 0
        for poly in geom.geoms:
            total += _unique_vertex_count(poly) or 0
        return total
    return get_num_coordinates(geom)


def _build_summary_batch(
    batch: pa.RecordBatch,
    counts: FileCounts,
    schema: pa.Schema,
    convert_id_to_uuid: bool,
    geod: Geod,
) -> pa.RecordBatch:
    if batch.num_rows == 0:
        return pa.RecordBatch.from_arrays([], schema=schema)

    bbox_array: pa.StructArray = batch.column(0)
    sources_array = batch.column(1)
    geometry_array = batch.column(2)
    height_array_in = batch.column(3)

    xmin = bbox_array.field("xmin").to_numpy(zero_copy_only=False)
    xmax = bbox_array.field("xmax").to_numpy(zero_copy_only=False)
    ymin = bbox_array.field("ymin").to_numpy(zero_copy_only=False)
    ymax = bbox_array.field("ymax").to_numpy(zero_copy_only=False)

    x_center = xmin + (xmax - xmin) / 2.0
    y_center = ymin + (ymax - ymin) / 2.0

    point_geometries = shapely_points(x_center, y_center)
    point_wkb = shapely_to_wkb(point_geometries, hex=False)
    point_array = pa.array(point_wkb, type=schema.field(0).type)

    polygon_wkb = geometry_array.to_numpy(zero_copy_only=False)
    polygon_geoms = shapely_from_wkb(polygon_wkb)

    area_sqm: list[int | None] = []
    vertex_count: list[int | None] = []
    for geom in polygon_geoms:
        if geom is None:
            area_sqm.append(None)
            vertex_count.append(None)
            continue
        area = abs(geod.geometry_area_perimeter(geom)[0])
        area_sqm.append(int(round(area)))
        vertex = _unique_vertex_count(geom)
        vertex_count.append(None if vertex is None else int(vertex))

    s1_list: list[int] = []
    s2_list: list[int] = []
    height_values_raw = height_array_in.to_pylist()
    height_values: list[int | None] = []
    for h in height_values_raw:
        if h is None:
            height_values.append(None)
        else:
            height_values.append(int(round(h)))

    for value in sources_array.to_pylist():
        s1, s2, combo_ids = _source_ids(value)
        s1_list.append(s1)
        s2_list.append(s2)
        counts.rows += 1
        for cid in combo_ids:
            counts.dataset_counts[ID_TO_NAME.get(cid, "other")] += 1
        counts.combination_counts[_combo_label(combo_ids)] += 1

    s1_array = pa.array(s1_list, type=schema.field(2).type)
    s2_array = pa.array(s2_list, type=schema.field(3).type)

    return pa.RecordBatch.from_arrays(
        [
            point_array,
            s1_array,
            s2_array,
            pa.array(area_sqm, type=schema.field(3).type),
            pa.array(vertex_count, type=schema.field(4).type),
            pa.array(height_values, type=schema.field(5).type),
        ],
        schema=schema,
    )


def _summary_batches(
    parquet_file: pq.ParquetFile,
    batch_rows: int,
    counts: FileCounts,
    schema: pa.Schema,
    convert_id_to_uuid: bool,
    geod: Geod,
):
    for row_group_idx in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(
            row_group_idx, columns=["bbox", "sources", "geometry", "height"]
        )
        for batch in table.to_batches(max_chunksize=batch_rows):
            yield _build_summary_batch(batch, counts, schema, convert_id_to_uuid, geod)


def _write_empty_summary(output_path: Path, schema: pa.Schema) -> None:
    empty_batch = pa.RecordBatch.from_arrays(
        [
            pa.array([], type=schema.field(0).type),
            pa.array([], type=schema.field(1).type),
            pa.array([], type=schema.field(2).type),
            pa.array([], type=schema.field(3).type),
            pa.array([], type=schema.field(4).type),
            pa.array([], type=schema.field(5).type),
        ],
        schema=schema,
    )
    empty_table = pa.Table.from_batches([empty_batch])
    pq.write_table(empty_table, output_path, compression="zstd")


def _summarize_file(task: Task) -> tuple[Path, str, FileCounts]:
    input_path, output_path = task.input_path, task.output_path
    counts = FileCounts(dataset_counts=Counter(), combination_counts=Counter())

    if task.skip_existing and output_path.exists():
        return input_path, "skipped (exists)", counts

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        parquet_file = pq.ParquetFile(input_path)
    except Exception as exc:  # pragma: no cover - defensive logging
        return input_path, f"failed to open ({exc})", counts

    convert_id_to_uuid = False
    summary_schema = _make_summary_schema()

    writer: pq.ParquetWriter | None = None
    total_rows = 0
    next_progress = LOG_PROGRESS_EVERY
    try:
        for batch in _summary_batches(
            parquet_file,
            task.batch_rows,
            counts,
            summary_schema,
            convert_id_to_uuid,
            GEOD,
        ):
            if batch.num_rows == 0:
                continue
            if writer is None:
                writer = pq.ParquetWriter(output_path, summary_schema, compression="zstd")
            writer.write_batch(batch)
            total_rows += batch.num_rows
            while total_rows >= next_progress:
                LOGGER.info("%s: processed %s rows", input_path.name, next_progress)
                next_progress += LOG_PROGRESS_EVERY
    finally:
        if writer is not None:
            writer.close()

    if total_rows == 0:
        _write_empty_summary(output_path, summary_schema)
        return input_path, "wrote empty summary", counts

    return input_path, f"wrote {total_rows} rows", counts


def _collect_input_files(input_dir: Path, max_files: int | None):
    parquet_files = sorted(input_dir.rglob("*.parquet"))
    if max_files is not None:
        parquet_files = parquet_files[: max(0, max_files)]
    return parquet_files


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "buildings_dir",
        type=Path,
        help="Path to the directory that contains the buildings GeoParquet files.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Where to write summary GeoParquet files (default: {DEFAULT_RESULTS_DIR}).",
    )
    parser.add_argument(
        "--log-json",
        type=Path,
        default=None,
        help="Optional path to write JSON log with dataset/combination counts.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="Maximum number of parallel workers to use.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit on how many GeoParquet files to process.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        default=True,
        help="Recreate summary files even if they already exist.",
    )
    parser.add_argument(
        "--batch-rows",
        type=int,
        default=DEFAULT_BATCH_ROWS,
        help="Maximum number of rows to hold in memory per chunk when summarizing a file.",
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
    )


def _merge_counts(total: Counter[str], increment: Counter[str]) -> None:
    total.update(increment)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results_dir = args.results_dir.expanduser()
    results_dir.mkdir(parents=True, exist_ok=True)

    log_path = results_dir / LOG_FILENAME
    configure_logging(args.log_level, log_path)
    LOGGER.debug("Writing log to %s", log_path)

    buildings_dir = args.buildings_dir.expanduser()
    if not buildings_dir.exists():
        LOGGER.error("Input directory %s does not exist.", buildings_dir)
        return 1

    input_files = _collect_input_files(buildings_dir, args.max_files)
    if not input_files:
        LOGGER.warning("No GeoParquet files found under %s", buildings_dir)
        return 0

    LOGGER.info(
        "Processing %s file(s) into %s using %s worker(s)",
        len(input_files),
        results_dir,
        args.max_workers,
    )

    tasks = [
        Task(
            input_path=path,
            output_path=_derive_output_path(path, results_dir),
            skip_existing=args.skip_existing,
            batch_rows=args.batch_rows,
        )
        for path in input_files
    ]

    start_time = time.perf_counter()
    per_file_counts: dict[str, FileCounts] = {}

    if args.max_workers == 1:
        for task in tasks:
            _, message, counts = _summarize_file(task)
            LOGGER.info("%s: %s", task.input_path.name, message)
            per_file_counts[task.input_path.name] = counts
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            for task, result in zip(tasks, executor.map(_summarize_file, tasks)):
                _, message, counts = result
                LOGGER.info("%s: %s", task.input_path.name, message)
                per_file_counts[task.input_path.name] = counts

    elapsed = time.perf_counter() - start_time
    LOGGER.info("Finished all files in %.2f seconds", elapsed)

    if args.log_json:
        # Aggregate counts
        total_dataset_counts: Counter[str] = Counter()
        total_combo_counts: Counter[str] = Counter()
        for counts in per_file_counts.values():
            _merge_counts(total_dataset_counts, counts.dataset_counts)
            _merge_counts(total_combo_counts, counts.combination_counts)

        log_data = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "input_dir": str(buildings_dir),
            "results_dir": str(results_dir),
            "files_processed": len(per_file_counts),
            "elapsed_seconds": elapsed,
            "dataset_counts": dict(total_dataset_counts),
            "combination_counts": dict(total_combo_counts),
            "per_file": {
                fname: {
                    "rows": counts.rows,
                    "dataset_counts": dict(counts.dataset_counts),
                    "combination_counts": dict(counts.combination_counts),
                }
                for fname, counts in sorted(per_file_counts.items())
            },
        }
        log_json_path = args.log_json.expanduser()
        log_json_path.parent.mkdir(parents=True, exist_ok=True)
        with log_json_path.open("w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2)
        LOGGER.info("Wrote JSON log to %s", log_json_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
