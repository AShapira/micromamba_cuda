#!/usr/bin/env python3
"""
Aggregate building dataset counts per region using a Shapely STRtree on regions.

For each summary GeoParquet file (geometry point + s1/s2 ids + area/vertex stats), the script:
1) Streams the file in large batches.
2) Uses a region STRtree (built once per worker) to find candidate polygons.
3) Counts buildings by region and dataset (s1 only; s2 ignored).
4) Aggregates area_sqm and vertex_count per region and dataset.
5) Writes a per-file GeoParquet with all region columns plus dataset count/sum columns.

After all per-file outputs exist, a final merged GeoParquet can be produced by
row-wise summing the per-file counts. Skips processing for files whose outputs
already exist, making the pipeline restartable.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from shapely import from_wkb as shapely_from_wkb
from shapely.prepared import prep as shapely_prep
from shapely.strtree import STRtree

LOGGER = logging.getLogger("region_dataset_counts_strtree")
LOG_PROGRESS_EVERY = 1_000_000

# Dataset ID mapping (lowercased keys) based on summarize_buildings_by_source.py
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
    import re
    import unicodedata

    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^0-9a-z]+", "_", ascii_name.lower()).strip("_")
    return slug or "dataset"


def _build_dataset_columns() -> tuple[list[DatasetColumn], dict[int, int], np.ndarray]:
    columns: list[DatasetColumn] = [DatasetColumn(dataset_id=UNKNOWN_ID, column_name="unknown", label="unknown")]
    for name, ds_id in sorted(DATASET_ID_MAP.items(), key=lambda item: item[1]):
        columns.append(DatasetColumn(dataset_id=ds_id, column_name=_slugify(name), label=name))
    idx_map = {col.dataset_id: idx for idx, col in enumerate(columns)}

    # Fast lookup for int8 s1 values: 256-length array indexed by value+128
    lookup = np.full(256, idx_map[UNKNOWN_ID], dtype=np.int16)
    for ds_id, idx in idx_map.items():
        if -128 <= ds_id <= 127:
            lookup[ds_id + 128] = idx
    return columns, idx_map, lookup


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


def load_regions(regions_path: Path) -> tuple[pa.Table, np.ndarray, np.ndarray, list[str]]:
    table = pq.read_table(regions_path)
    if "geometry" not in table.column_names:
        raise ValueError("Regions parquet must contain a 'geometry' column (WKB).")

    geometries = shapely_from_wkb(table.column("geometry").to_numpy())
    region_geoms = np.asarray(geometries)
    region_idx = np.arange(len(region_geoms), dtype=np.int32)
    return table, region_geoms, region_idx, table.column_names


@dataclass(frozen=True)
class Task:
    input_path: Path
    output_path: Path
    skip_existing: bool
    row_limit: int | None
    batch_size: int


WORKER_TREE: STRtree | None = None
WORKER_REGION_IDX: np.ndarray | None = None
WORKER_DATASET_LOOKUP: np.ndarray | None = None
WORKER_DATASET_COLUMNS: list[DatasetColumn] = []
WORKER_PREPARED: list | None = None


def _init_worker(region_geoms: np.ndarray, region_idx: np.ndarray, dataset_lookup: np.ndarray, dataset_cols: list[DatasetColumn]) -> None:
    global WORKER_TREE, WORKER_REGION_IDX, WORKER_DATASET_LOOKUP, WORKER_DATASET_COLUMNS, WORKER_PREPARED
    region_list = region_geoms.tolist()
    WORKER_TREE = STRtree(region_list)
    WORKER_REGION_IDX = region_idx
    WORKER_DATASET_LOOKUP = dataset_lookup
    WORKER_DATASET_COLUMNS = dataset_cols
    WORKER_PREPARED = [shapely_prep(g) for g in region_list]


def _process_summary(task: Task) -> tuple[str, str]:
    if WORKER_TREE is None or WORKER_REGION_IDX is None or WORKER_DATASET_LOOKUP is None:
        raise RuntimeError("Worker not initialized with region tree.")

    input_path, output_path = task.input_path, task.output_path
    if task.skip_existing and output_path.exists():
        return input_path.name, "skipped (exists)"

    pq_file = pq.ParquetFile(input_path)
    region_count = len(WORKER_REGION_IDX)
    dataset_count = len(WORKER_DATASET_COLUMNS)
    counts = np.zeros((region_count, dataset_count), dtype=np.int64)
    area_sums = np.zeros_like(counts, dtype=np.int64)
    vertex_sums = np.zeros_like(counts, dtype=np.int64)

    processed = 0
    next_progress = LOG_PROGRESS_EVERY
    for batch in pq_file.iter_batches(columns=["geometry", "s1", "area_sqm", "vertex_count"], batch_size=task.batch_size):
        geom_wkb = batch.column(0).to_numpy(zero_copy_only=False)
        points = shapely_from_wkb(geom_wkb)

        s1_array = batch.column(1)
        if s1_array.null_count:
            zero_array = pa.array(np.zeros(len(s1_array), dtype=np.int8), type=pa.int8())
            s1_array = pc.coalesce(s1_array, zero_array)
        s1_np = s1_array.to_numpy(zero_copy_only=False)

        area_array = batch.column(2)
        if area_array.null_count:
            area_array = pc.fill_null(area_array, 0)
        area_np = area_array.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)

        vertex_array = batch.column(3)
        if vertex_array.null_count:
            vertex_array = pc.fill_null(vertex_array, 0)
        vertex_np = vertex_array.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)

        if len(points) and WORKER_TREE is not None and WORKER_PREPARED is not None:
            for idx, pt in enumerate(points):
                candidate_idx = WORKER_TREE.query(pt)
                if len(candidate_idx) == 0:
                    continue
                ds_idx = WORKER_DATASET_LOOKUP[int(s1_np[idx]) + 128]
                for ridx in candidate_idx:
                    if WORKER_PREPARED[ridx].covers(pt):
                        counts[ridx, ds_idx] += 1
                        area_sums[ridx, ds_idx] += area_np[idx]
                        vertex_sums[ridx, ds_idx] += vertex_np[idx]

        processed += batch.num_rows
        while processed >= next_progress:
            LOGGER.info("%s: processed %s rows", input_path.name, next_progress)
            next_progress += LOG_PROGRESS_EVERY
        if task.row_limit and processed >= task.row_limit:
            break

    # Write per-file counts as Arrow Table (geometry retained for GeoParquet friendliness)
    arrays = []
    names = []
    # geometry + all region columns are written later by caller
    for idx, col in enumerate(WORKER_DATASET_COLUMNS):
        arrays.append(pa.array(counts[:, idx], type=pa.int64()))
        names.append(col.column_name)
        arrays.append(pa.array(area_sums[:, idx], type=pa.int64()))
        names.append(f"area_{col.column_name}")
        arrays.append(pa.array(vertex_sums[:, idx], type=pa.int64()))
        names.append(f"vertices_{col.column_name}")

    per_file_table = pa.Table.from_arrays(arrays, names=names)
    pq.write_table(per_file_table, output_path, compression="ZSTD")
    return input_path.name, f"wrote {processed} rows"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_dir", type=Path, help="Directory containing building summary GeoParquets (geometry + s1).")
    parser.add_argument("regions_path", type=Path, help="Region GeoParquet (region_area-*.parquet).")
    parser.add_argument("--output-dir", type=Path, default=Path("data/results/region_dataset_counts_strtree"), help="Per-file output directory (GeoParquet).")
    parser.add_argument("--final-output", type=Path, default=None, help="Optional final merged GeoParquet path.")
    parser.add_argument("--max-workers", type=int, default=max(1, (os.cpu_count() or 2) // 2), help="Number of parallel workers.")
    parser.add_argument("--batch-size", type=int, default=400_000, help="Rows per streamed batch.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional limit on summary files (for testing).")
    parser.add_argument("--row-limit", type=int, default=None, help="Optional max rows to process per file (for benchmarking).")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity.")
    parser.add_argument("--no-skip-existing", action="store_false", dest="skip_existing", default=True, help="Reprocess outputs even if they exist.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    summary_dir = args.summary_dir.expanduser()
    regions_path = args.regions_path.expanduser()
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not summary_dir.exists():
        raise FileNotFoundError(f"Summary directory {summary_dir} does not exist.")
    if not regions_path.exists():
        raise FileNotFoundError(f"Regions file {regions_path} does not exist.")

    summary_files = sorted(path for path in summary_dir.glob("*.parquet") if path.is_file())
    if args.max_files is not None:
        summary_files = summary_files[: max(0, args.max_files)]
    if not summary_files:
        raise FileNotFoundError(f"No summary parquet files found in {summary_dir}.")

    regions_table, region_geoms, region_idx, region_columns = load_regions(regions_path)
    dataset_columns, dataset_idx_map, dataset_lookup = _build_dataset_columns()
    tasks = [
        Task(
            input_path=path,
            output_path=output_dir / f"{path.stem}_region_counts.parquet",
            skip_existing=args.skip_existing,
            row_limit=args.row_limit,
            batch_size=args.batch_size,
        )
        for path in summary_files
    ]

    log_path = output_dir / "aggregate_region_dataset_counts_strtree.log"
    configure_logging(args.log_level, log_path)
    LOGGER.info("Writing log to %s", log_path)
    worker_count = max(1, min(args.max_workers, len(tasks)))
    LOGGER.info("Processing %s file(s) with %s worker(s)", len(tasks), worker_count)

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_init_worker,
        initargs=(region_geoms, region_idx, dataset_lookup, dataset_columns),
    ) as executor:
        futures = {executor.submit(_process_summary, task): task for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            name, message = future.result()
            LOGGER.info("%s: %s", name, message)

    if args.final_output:
        LOGGER.info("Merging per-file outputs into %s", args.final_output)
        # Start with zeros; shape = regions x dataset columns
        total_counts = np.zeros((len(region_idx), len(dataset_columns)), dtype=np.int64)
        total_area = np.zeros_like(total_counts, dtype=np.int64)
        total_vertices = np.zeros_like(total_counts, dtype=np.int64)
        for task in tasks:
            out_path = task.output_path
            if not out_path.exists():
                LOGGER.warning("Skipping missing per-file output %s", out_path)
                continue
            cols_to_read = []
            for col in dataset_columns:
                cols_to_read.append(col.column_name)
                cols_to_read.append(f"area_{col.column_name}")
                cols_to_read.append(f"vertices_{col.column_name}")
            table = pq.read_table(out_path, columns=cols_to_read)
            counts = np.vstack([np.array(table.column(col.column_name)) for col in dataset_columns]).T
            areas = np.vstack([np.array(table.column(f"area_{col.column_name}")) for col in dataset_columns]).T
            vertices = np.vstack([np.array(table.column(f"vertices_{col.column_name}")) for col in dataset_columns]).T
            total_counts += counts
            total_area += areas
            total_vertices += vertices

        # Build final table: all region columns + dataset count/sum columns
        final_cols = {name: regions_table.column(name) for name in region_columns}
        for idx, col in enumerate(dataset_columns):
            final_cols[col.column_name] = pa.array(total_counts[:, idx], type=pa.int64())
            final_cols[f"area_{col.column_name}"] = pa.array(total_area[:, idx], type=pa.int64())
            final_cols[f"vertices_{col.column_name}"] = pa.array(total_vertices[:, idx], type=pa.int64())
        final_table = pa.table(final_cols, schema=None)
        pq.write_table(final_table, args.final_output.expanduser(), compression="ZSTD")
        LOGGER.info("Wrote %s", args.final_output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
