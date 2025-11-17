#!/usr/bin/env python3
"""
Summarize Overture buildings GeoParquet files into lightweight point layers.

For every GeoParquet in the input directory a summary GeoParquet is written to
data/results/buildings_osm_summary (or a user supplied directory). Each summary
row corresponds to one building row and contains:
* `geometry`: Point geometry representing the center of the original bbox
* `is_osm`: True when the building sources include the OpenStreetMap dataset
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence
import pyarrow as pa
import pyarrow.parquet as pq
from shapely import points as shapely_points
from shapely import to_wkb as shapely_to_wkb


LOGGER = logging.getLogger("buildings_summary")
DEFAULT_RESULTS_DIR = Path("data/results/buildings_osm_summary")
LOG_FILENAME = "buildings_summary.log"
OUTPUT_BASENAME_RE = re.compile(r"^(part-\d+)")
DEFAULT_BATCH_ROWS = 200_000
LOG_PROGRESS_EVERY = 1_000_000


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


SUMMARY_SCHEMA = pa.schema(
    [
        pa.field("geometry", pa.binary()),
        pa.field("is_osm", pa.bool_()),
    ]
).with_metadata(_build_geo_metadata())


@dataclass(frozen=True)
class Task:
    input_path: Path
    output_path: Path
    skip_existing: bool
    batch_rows: int


def _derive_output_path(input_path: Path, results_dir: Path) -> Path:
    match = OUTPUT_BASENAME_RE.match(input_path.name)
    if match:
        prefix = match.group(1)
    else:
        prefix = input_path.stem
    filename = f"{prefix}-summary.parquet"
    return results_dir / filename


def _has_osm_source(source_list: list | None) -> bool:
    if not source_list:
        return False
    for entry in source_list:
        dataset = entry.get("dataset") if isinstance(entry, dict) else None
        if dataset and dataset.lower() == "openstreetmap":
            return True
    return False


def _empty_batch() -> pa.RecordBatch:
    return pa.RecordBatch.from_arrays(
        [pa.array([], type=SUMMARY_SCHEMA.field(i).type) for i in range(len(SUMMARY_SCHEMA))],
        schema=SUMMARY_SCHEMA,
    )


def _build_summary_batch(batch: pa.RecordBatch) -> pa.RecordBatch:
    if batch.num_rows == 0:
        return _empty_batch()

    bbox_array: pa.StructArray = batch.column(0)
    sources_array: pa.ListArray = batch.column(1)

    xmin = bbox_array.field("xmin").to_numpy(zero_copy_only=False)
    xmax = bbox_array.field("xmax").to_numpy(zero_copy_only=False)
    ymin = bbox_array.field("ymin").to_numpy(zero_copy_only=False)
    ymax = bbox_array.field("ymax").to_numpy(zero_copy_only=False)

    x_center = xmin + (xmax - xmin) / 2.0
    y_center = ymin + (ymax - ymin) / 2.0

    point_geometries = shapely_points(x_center, y_center)
    geometry_wkb = shapely_to_wkb(point_geometries, hex=False)
    geometry_array = pa.array(geometry_wkb, type=SUMMARY_SCHEMA.field(0).type)

    is_osm_flags = [_has_osm_source(value) for value in sources_array.to_pylist()]
    osm_array = pa.array(is_osm_flags, type=SUMMARY_SCHEMA.field(1).type)

    return pa.RecordBatch.from_arrays([geometry_array, osm_array], schema=SUMMARY_SCHEMA)


def _summary_batches(parquet_file: pq.ParquetFile, batch_rows: int) -> Iterable[pa.RecordBatch]:
    for row_group_idx in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(row_group_idx, columns=["bbox", "sources"])
        for batch in table.to_batches(max_chunksize=batch_rows):
            yield _build_summary_batch(batch)


def _write_empty_summary(output_path: Path) -> None:
    empty_table = pa.Table.from_batches([_empty_batch()])
    pq.write_table(empty_table, output_path, compression="zstd")


def _summarize_file(task: Task) -> tuple[Path, str]:
    input_path, output_path = task.input_path, task.output_path
    if task.skip_existing and output_path.exists():
        LOGGER.info("Skipping existing summary for %s", input_path.name)
        return input_path, "skipped (exists)"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Starting %s", input_path.name)

    try:
        parquet_file = pq.ParquetFile(input_path)
    except Exception as exc:  # pragma: no cover - defensive logging
        return input_path, f"failed to open ({exc})"

    writer: pq.ParquetWriter | None = None
    total_rows = 0
    next_progress = LOG_PROGRESS_EVERY
    try:
        for batch in _summary_batches(parquet_file, task.batch_rows):
            if batch.num_rows == 0:
                continue
            if writer is None:
                writer = pq.ParquetWriter(output_path, SUMMARY_SCHEMA, compression="zstd")
            writer.write_batch(batch)
            total_rows += batch.num_rows
            while total_rows >= next_progress:
                LOGGER.info("%s: processed %s rows", input_path.name, next_progress)
                next_progress += LOG_PROGRESS_EVERY
    finally:
        if writer is not None:
            writer.close()

    if total_rows == 0:
        _write_empty_summary(output_path)
        LOGGER.info("Finished %s: wrote empty summary", input_path.name)
        return input_path, "wrote empty summary"

    LOGGER.info("Finished %s: wrote %s rows", input_path.name, total_rows)
    return input_path, f"wrote {total_rows} rows"


def _collect_input_files(input_dir: Path, max_files: int | None) -> List[Path]:
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


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    buildings_dir = args.buildings_dir.expanduser()
    results_dir = args.results_dir.expanduser()
    results_dir.mkdir(parents=True, exist_ok=True)

    log_path = results_dir / LOG_FILENAME
    configure_logging(args.log_level, log_path)
    LOGGER.debug("Writing log to %s", log_path)

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

    if args.max_workers == 1:
        for task in tasks:
            _, message = _summarize_file(task)
            LOGGER.info("%s: %s", task.input_path.name, message)
        return 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        for task, message in zip(tasks, executor.map(_summarize_file, tasks)):
            _, status = message
            LOGGER.info("%s: %s", task.input_path.name, status)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
