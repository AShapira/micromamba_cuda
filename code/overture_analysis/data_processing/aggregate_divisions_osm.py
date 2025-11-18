#!/usr/bin/env python3
"""
Aggregate OSM/non-OSM building counts per division polygon.

Each worker processes one buildings summary GeoParquet file, tallies how many
points fall inside every division geometry, and returns temporary counts that
are merged into the final GeoDataFrame.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
from pathlib import Path
from typing import Iterable, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from shapely import STRtree, contains, from_wkb, to_wkb
from shapely.geometry import box

LOGGER = logging.getLogger("divisions_summary")
LOG_PROGRESS_EVERY = 1_000_000
DEFAULT_BATCH_ROWS = 200_000

WORKER_TEMPLATE_GDF: gpd.GeoDataFrame | None = None
WORKER_GEOMS = None
WORKER_BBOX_TREE: STRtree | None = None
WORKER_XMIN = None
WORKER_XMAX = None
WORKER_YMIN = None
WORKER_YMAX = None
WORKER_BATCH_ROWS = DEFAULT_BATCH_ROWS


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "summary_dir",
        type=Path,
        help="Directory containing buildings summary GeoParquet files.",
    )
    parser.add_argument(
        "divisions_path",
        type=Path,
        help="GeoParquet file with division polygons (same schema as country_area).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Number of worker processes to use.",
    )
    parser.add_argument(
        "--batch-rows",
        type=int,
        default=DEFAULT_BATCH_ROWS,
        help="Number of summary rows to load per batch from each file.",
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
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def _prepare_bbox_columns(gdf: gpd.GeoDataFrame) -> None:
    bbox_series = gdf.get("bbox")
    if bbox_series is None:
        bounds = gdf.geometry.bounds
        gdf["_bbox_xmin"] = bounds["minx"]
        gdf["_bbox_xmax"] = bounds["maxx"]
        gdf["_bbox_ymin"] = bounds["miny"]
        gdf["_bbox_ymax"] = bounds["maxy"]
        return

    gdf["_bbox_xmin"] = bbox_series.apply(lambda b: b.get("xmin") if isinstance(b, dict) else np.nan)
    gdf["_bbox_xmax"] = bbox_series.apply(lambda b: b.get("xmax") if isinstance(b, dict) else np.nan)
    gdf["_bbox_ymin"] = bbox_series.apply(lambda b: b.get("ymin") if isinstance(b, dict) else np.nan)
    gdf["_bbox_ymax"] = bbox_series.apply(lambda b: b.get("ymax") if isinstance(b, dict) else np.nan)

    bounds = gdf.geometry.bounds
    for col, bound_key in [
        ("_bbox_xmin", "minx"),
        ("_bbox_xmax", "maxx"),
        ("_bbox_ymin", "miny"),
        ("_bbox_ymax", "maxy"),
    ]:
        gdf[col] = gdf[col].fillna(bounds[bound_key])


def load_divisions(divisions_path: Path) -> tuple[gpd.GeoDataFrame, dict]:
    table = pq.read_table(divisions_path)
    pdf = table.to_pandas()
    if "geometry" not in pdf:
        raise ValueError("Divisions file must include a geometry column.")

    geometry = from_wkb(pdf["geometry"].to_numpy())
    gdf = gpd.GeoDataFrame(
        pdf.drop(columns=["geometry"]),
        geometry=gpd.GeoSeries(geometry, crs="EPSG:4326"),
        copy=False,
    )
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)

    gdf["osm_count"] = 0
    gdf["non_osm_count"] = 0

    _prepare_bbox_columns(gdf)

    payload = {
        "geometry_wkb": list(to_wkb(gdf.geometry, hex=False)),
        "bbox_xmin": gdf["_bbox_xmin"].to_numpy(dtype=float).tolist(),
        "bbox_xmax": gdf["_bbox_xmax"].to_numpy(dtype=float).tolist(),
        "bbox_ymin": gdf["_bbox_ymin"].to_numpy(dtype=float).tolist(),
        "bbox_ymax": gdf["_bbox_ymax"].to_numpy(dtype=float).tolist(),
    }
    return gdf, payload


def _init_worker(payload: dict, log_path: Path, log_level: str, batch_rows: int) -> None:
    configure_logging(log_level, log_path)

    global WORKER_GEOMS, WORKER_TEMPLATE_GDF, WORKER_BBOX_TREE
    global WORKER_XMIN, WORKER_XMAX, WORKER_YMIN, WORKER_YMAX, WORKER_BATCH_ROWS

    WORKER_GEOMS = from_wkb(np.array(payload["geometry_wkb"], dtype=object))
    WORKER_XMIN = np.array(payload["bbox_xmin"], dtype=float)
    WORKER_XMAX = np.array(payload["bbox_xmax"], dtype=float)
    WORKER_YMIN = np.array(payload["bbox_ymin"], dtype=float)
    WORKER_YMAX = np.array(payload["bbox_ymax"], dtype=float)
    WORKER_BATCH_ROWS = batch_rows

    # Build bounding boxes from the provided bbox columns for fast spatial filtering.
    bbox_geoms = [
        box(xmin, ymin, xmax, ymax)
        for xmin, xmax, ymin, ymax in zip(WORKER_XMIN, WORKER_XMAX, WORKER_YMIN, WORKER_YMAX)
    ]
    WORKER_BBOX_TREE = STRtree(bbox_geoms)

    template_df = pd.DataFrame(
        {
            "_bbox_xmin": WORKER_XMIN,
            "_bbox_xmax": WORKER_XMAX,
            "_bbox_ymin": WORKER_YMIN,
            "_bbox_ymax": WORKER_YMAX,
        }
    )
    WORKER_TEMPLATE_GDF = gpd.GeoDataFrame(template_df, geometry=gpd.GeoSeries(WORKER_GEOMS, crs="EPSG:4326"))


def _iter_summary_batches(path: Path) -> Iterable[tuple[np.ndarray, np.ndarray, int]]:
    parquet = pq.ParquetFile(path)
    for row_group_idx in range(parquet.num_row_groups):
        table = parquet.read_row_group(row_group_idx, columns=["geometry", "is_osm"])
        for batch in table.to_batches(max_chunksize=WORKER_BATCH_ROWS):
            geom_data = batch.column(0).to_numpy(zero_copy_only=False)
            osm_data = batch.column(1).to_numpy(zero_copy_only=False)
            yield geom_data, osm_data, batch.num_rows


def _process_summary_file(path: Path) -> tuple[str, np.ndarray, np.ndarray]:
    if (
        WORKER_TEMPLATE_GDF is None
        or WORKER_GEOMS is None
        or WORKER_BBOX_TREE is None
        or WORKER_XMIN is None
        or WORKER_XMAX is None
        or WORKER_YMIN is None
        or WORKER_YMAX is None
    ):
        raise RuntimeError("Worker state not initialized.")

    logger = logging.getLogger("divisions_summary")
    summary_name = path.name
    logger.info("Starting %s", summary_name)

    temp_gdf = WORKER_TEMPLATE_GDF.copy()
    temp_gdf["osm_count"] = 0
    temp_gdf["non_osm_count"] = 0
    osm_counts = temp_gdf["osm_count"].to_numpy(dtype=np.int64, copy=False)
    non_osm_counts = temp_gdf["non_osm_count"].to_numpy(dtype=np.int64, copy=False)

    processed_rows = 0
    next_progress = LOG_PROGRESS_EVERY

    for geom_chunk, osm_chunk, reported_rows in _iter_summary_batches(path):
        if len(geom_chunk) == 0:
            processed_rows += reported_rows
            continue

        geom_array = np.asarray(geom_chunk)
        osm_array = np.asarray(osm_chunk)

        valid_mask = np.fromiter((g is not None for g in geom_array), dtype=bool, count=len(geom_array))
        if not valid_mask.all():
            geom_array = geom_array[valid_mask]
            osm_array = osm_array[valid_mask]

        if geom_array.size == 0:
            processed_rows += reported_rows
            continue

        points = from_wkb(geom_array)
        osm_flags = (
            osm_array.astype(bool)
            if osm_array.dtype == bool
            else np.array([bool(value) for value in osm_array], dtype=bool)
        )

        candidate_pairs = WORKER_BBOX_TREE.query(points)
        if candidate_pairs.size:
            point_indexes, division_indexes = candidate_pairs
            matches = contains(WORKER_GEOMS[division_indexes], points[point_indexes])
            if np.any(matches):
                matched_divisions = division_indexes[matches]
                matched_points = point_indexes[matches]
                for div_idx, point_idx in zip(matched_divisions, matched_points):
                    if osm_flags[point_idx]:
                        osm_counts[div_idx] += 1
                    else:
                        non_osm_counts[div_idx] += 1

        processed_rows += reported_rows
        while processed_rows >= next_progress:
            logger.info("%s: processed %s rows", summary_name, next_progress)
            next_progress += LOG_PROGRESS_EVERY

    logger.info("%s: finished %s rows", summary_name, processed_rows)
    return summary_name, osm_counts.copy(), non_osm_counts.copy()


def _aggregate_counts(
    summary_files: list[Path],
    payload: dict,
    log_path: Path,
    log_level: str,
    max_workers: int,
    batch_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    worker_count = max(1, min(max_workers, len(summary_files)))
    LOGGER.info("Processing %s file(s) with %s worker(s)", len(summary_files), worker_count)

    osm_total: np.ndarray | None = None
    non_osm_total: np.ndarray | None = None

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_init_worker,
        initargs=(payload, log_path, log_level, batch_rows),
    ) as executor:
        future_map = {executor.submit(_process_summary_file, path): path for path in summary_files}
        for future in concurrent.futures.as_completed(future_map):
            summary_name, osm_counts, non_counts = future.result()
            LOGGER.info("Merged results for %s", summary_name)

            if osm_total is None:
                osm_total = osm_counts
                non_osm_total = non_counts
            else:
                osm_total += osm_counts
                non_osm_total += non_counts

    assert osm_total is not None and non_osm_total is not None
    return osm_total, non_osm_total


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    summary_dir = args.summary_dir.expanduser()
    divisions_path = args.divisions_path.expanduser()
    if not summary_dir.exists():
        raise FileNotFoundError(f"Summary directory {summary_dir} does not exist.")
    if not divisions_path.exists():
        raise FileNotFoundError(f"Divisions file {divisions_path} does not exist.")

    log_path = divisions_path.parent / f"{divisions_path.stem}_osm_summary.log"
    configure_logging(args.log_level, log_path)
    LOGGER.debug("Logging to %s", log_path)

    divisions_gdf, payload = load_divisions(divisions_path)
    summary_files = sorted(p for p in summary_dir.glob("*.parquet") if p.is_file())
    if not summary_files:
        LOGGER.warning("No summary files found in %s", summary_dir)
        return 0

    osm_counts, non_osm_counts = _aggregate_counts(
        summary_files, payload, log_path, args.log_level, args.max_workers, args.batch_rows
    )

    divisions_gdf["osm_count"] = osm_counts
    divisions_gdf["non_osm_count"] = non_osm_counts

    drop_cols = [col for col in divisions_gdf.columns if col.startswith("_bbox_")]
    if drop_cols:
        divisions_gdf = divisions_gdf.drop(columns=drop_cols)

    output_path = divisions_path.with_name(f"{divisions_path.stem}_osm_summary{divisions_path.suffix}")
    divisions_gdf.to_parquet(output_path, index=False)
    LOGGER.info("Wrote %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
