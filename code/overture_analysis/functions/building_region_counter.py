"""Aggregate building counts per dataset for each region geometry."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from shapely.geometry.base import BaseGeometry


def summarise_buildings_by_source(
    regions_gdf: gpd.GeoDataFrame,
    *,
    buildings_base_path: str | Path,
    batch_size: int = 5000,
    logger: logging.Logger | None = None,
    filter_bbox: tuple[float, float, float, float] | None = None,
    processing_dir: str | Path | None = None,
    max_workers: int | None = None,
) -> list[Path]:
    """Process building files in parallel and emit per-file region summaries.

    The workflow assigns one worker thread per GeoParquet file and writes per-file
    GeoParquet outputs under ``processing_dir``.  Use ``merge_region_summaries`` to
    combine the resulting files into a single GeoDataFrame.

    Parameters
    ----------
    regions_gdf:
        GeoDataFrame containing the regions to summarise.  The geometry column must be
        in the same CRS as the buildings dataset (WGS84 / EPSG:4326).
    buildings_base_path:
        Directory pointing at ``theme=buildings/type=building`` GeoParquet files.
    batch_size:
        Number of rows to process at a time while streaming from Parquet.
    logger:
        Optional logger used for progress reporting.  When omitted, the module logger is
        used.
    filter_bbox:
        Optional bounding box (minx, miny, maxx, maxy) used to discard building files
        whose declared extents fall outside the regions of interest.
    processing_dir:
        Directory where each worker writes its per-file GeoParquet output.  Existing
        ``*.geoparquet`` files in this directory are removed before processing.
    max_workers:
        Optional upper bound for the number of worker threads.  Defaults to the number
        of candidate files (bounded by ``ThreadPoolExecutor`` defaults).
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if processing_dir is None:
        raise ValueError("processing_dir must be provided to store intermediate results.")

    log = logger or logging.getLogger(__name__)

    if regions_gdf.empty:
        log.info("regions_gdf is empty; returning original frame.")
        return regions_gdf.copy()

    if regions_gdf.crs is None:
        raise ValueError("regions_gdf must define a CRS that matches the buildings data.")

    base_path = Path(buildings_base_path).expanduser().resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"Buildings directory does not exist: {base_path}")

    processing_dir = Path(processing_dir).expanduser().resolve()
    processing_dir.mkdir(parents=True, exist_ok=True)
    for stale_file in processing_dir.glob("*.geoparquet"):
        try:
            stale_file.unlink()
        except FileNotFoundError:
            pass

    file_index, geometry_column = _load_file_index(base_path)
    if not file_index:
        raise FileNotFoundError(f"No GeoParquet files found under {base_path}")

    if filter_bbox is not None:
        filtered_files = [entry for entry in file_index if _bbox_intersects(entry.bbox, filter_bbox)]
        if not filtered_files:
            log.warning("No building files intersect the provided bounding box; returning original regions.")
            return regions_gdf.copy()
    else:
        filtered_files = file_index

    try:
        spatial_index = regions_gdf.sindex
    except Exception as exc:  # pragma: no cover - geopandas raises informative errors
        raise RuntimeError("Unable to build a spatial index for regions_gdf.") from exc

    if spatial_index is None:
        raise RuntimeError(
            "Spatial index unavailable. Install either pygeos or rtree for geopandas."
        )

    geometry_name = regions_gdf.geometry.name

    log.info(
        "Scanning %d GeoParquet files under %s%s",
        len(filtered_files),
        base_path,
        "" if filter_bbox is None else " constrained by bounding box",
    )

    if max_workers is None:
        max_workers = len(filtered_files) or 1
    else:
        max_workers = min(max_workers, len(filtered_files)) or 1
    if max_workers <= 0:
        raise ValueError("max_workers must be positive.")

    log.info("Dispatching %d worker thread(s) for %d GeoParquet files", max_workers, len(filtered_files))

    seen_datasets: set[str] = set()
    partial_paths: list[Path] = []

    def _process_file(entry: _FileEntry) -> tuple[Path, set[str]]:
        local_gdf = regions_gdf.copy(deep=True)
        local_seen: set[str] = set()
        file_rows = 0
        matches_in_file = 0
        next_log_threshold = 100_000

        log.info("Starting worker for %s", entry.path.name)
        parquet_file = pq.ParquetFile(entry.path)
        arrow_kwargs = {}
        if hasattr(pd, "ArrowDtype"):
            arrow_kwargs["types_mapper"] = pd.ArrowDtype

        for batch in parquet_file.iter_batches(columns=[geometry_column, "sources"], batch_size=batch_size):
            batch_df = batch.to_pandas(**arrow_kwargs)
            geometries = gpd.GeoSeries.from_wkb(
                batch_df[geometry_column], crs=regions_gdf.crs, name=geometry_name
            )

            for geom, sources in zip(geometries, batch_df["sources"]):
                file_rows += 1
                if file_rows >= next_log_threshold:
                    log.info("%s: processed %d buildings so far", entry.path.name, file_rows)
                    next_log_threshold += 100_000

                if geom is None or (isinstance(geom, BaseGeometry) and geom.is_empty):
                    continue

                dataset = _first_dataset_name(sources)
                if not dataset:
                    continue

                dataset = str(dataset)
                if dataset not in local_gdf.columns:
                    local_gdf[dataset] = 0
                local_seen.add(dataset)

                bounds = geom.bounds
                candidate_indexes = list(spatial_index.intersection(bounds))
                if not candidate_indexes:
                    continue

                candidate_regions = regions_gdf.iloc[candidate_indexes]
                match_mask = candidate_regions.contains(geom)
                if not match_mask.any():
                    continue

                for idx in candidate_regions.index[match_mask]:
                    local_gdf.at[idx, dataset] = int(local_gdf.at[idx, dataset]) + 1
                    matches_in_file += 1

        output_path = processing_dir / f"{entry.path.stem}_summary.geoparquet"
        local_gdf.to_parquet(output_path, index=False)
        log.info(
            "%s: completed %d buildings with %d matches; wrote %s",
            entry.path.name,
            file_rows,
            matches_in_file,
            output_path.name,
        )
        return output_path, local_seen

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_file, entry): entry for entry in filtered_files}
        for future in as_completed(futures):
            entry = futures[future]
            try:
                partial_path, datasets = future.result()
            except Exception as exc:  # pragma: no cover - propagate with context
                log.error("Failed to process %s", entry.path, exc_info=exc)
                raise
            partial_paths.append(partial_path)
            seen_datasets.update(datasets)

    log.info(
        "Completed scanning. Dataset coverage across workers: %s",
        ", ".join(sorted(seen_datasets)) or "none",
    )
    log.info("Generated %d partial GeoParquet files under %s", len(partial_paths), processing_dir)
    return partial_paths


def merge_region_summaries(
    regions_template: gpd.GeoDataFrame,
    summary_paths: Sequence[str | Path],
    *,
    logger: logging.Logger | None = None,
) -> gpd.GeoDataFrame:
    """Merge per-file region summaries into a single GeoDataFrame."""

    if regions_template.empty:
        raise ValueError("regions_template must contain at least one row.")

    paths = [Path(path).expanduser().resolve() for path in summary_paths]
    if not paths:
        raise ValueError("summary_paths must contain at least one GeoParquet file.")

    log = logger or logging.getLogger(__name__)
    aggregated_gdf = regions_template.copy()
    geometry_name = aggregated_gdf.geometry.name
    base_columns = {col for col in aggregated_gdf.columns if col != geometry_name}

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Summary file not found: {path}")
        partial_gdf = gpd.read_parquet(path)
        _accumulate_dataset_columns(
            aggregated_gdf=aggregated_gdf,
            partial_gdf=partial_gdf,
            geometry_name=geometry_name,
            base_columns=base_columns,
            summary_path=path,
        )
        log.info("Merged counts from %s", path.name)

    return aggregated_gdf


def _first_dataset_name(sources: Any) -> str | None:
    """Extract the dataset field from the first entry in ``sources``."""

    if sources is None:
        return None
    if isinstance(sources, float) and pd.isna(sources):
        return None

    iterable: Iterable[Any]

    if isinstance(sources, Mapping):
        iterable = [sources]
    elif isinstance(sources, str):
        return sources
    elif isinstance(sources, Sequence) and not isinstance(sources, (bytes, bytearray)):
        iterable = sources
    elif hasattr(sources, "tolist"):
        iterable = sources.tolist()
    else:
        return None

    for item in iterable:
        if item is None:
            continue
        if isinstance(item, Mapping):
            dataset = item.get("dataset")
        elif hasattr(item, "get"):
            try:
                dataset = item.get("dataset")
            except Exception:  # pragma: no cover - fall back to attribute lookup
                dataset = getattr(item, "dataset", None)
        else:
            dataset = getattr(item, "dataset", None)
        if dataset:
            return dataset
    return None


@dataclass(frozen=True)
class _FileEntry:
    path: Path
    bbox: tuple[float, float, float, float]


def _load_file_index(base_path: Path) -> tuple[list[_FileEntry], str]:
    geometry_column: str | None = None
    entries: list[_FileEntry] = []

    for file_path in sorted(base_path.rglob("*.parquet")):
        parquet_file = pq.ParquetFile(file_path)
        metadata = parquet_file.metadata.metadata or {}
        decoded = {
            (k.decode("utf-8") if isinstance(k, (bytes, bytearray)) else k): (
                v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else v
            )
            for k, v in metadata.items()
        }
        geo_payload = decoded.get("geo")
        if not geo_payload:
            continue
        geo_json = json.loads(geo_payload)
        file_geometry_column = geo_json.get("primary_column")
        if not file_geometry_column:
            raise ValueError(f"Primary geometry column missing in {file_path}")

        if geometry_column is None:
            geometry_column = file_geometry_column
        elif geometry_column != file_geometry_column:
            raise ValueError(
                "Multiple geometry columns detected; expected a single consistent column."
            )
        columns_meta = geo_json.get("columns", {})
        column_meta = columns_meta.get(geometry_column, {}) if columns_meta else {}
        bbox = column_meta.get("bbox")
        if not bbox or len(bbox) != 4:
            raise ValueError(f"Bounding box metadata missing for {file_path}")
        minx, miny, maxx, maxy = map(float, bbox)
        entries.append(_FileEntry(path=file_path, bbox=(minx, miny, maxx, maxy)))

    if geometry_column is None:
        raise ValueError("Unable to detect geometry column from GeoParquet metadata.")

    return entries, geometry_column


def _bbox_intersects(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    aminx, aminy, amaxx, amaxy = a
    bminx, bminy, bmaxx, bmaxy = b
    if amaxx < bminx or bmaxx < aminx:
        return False
    if amaxy < bminy or bmaxy < aminy:
        return False
    return True


def _accumulate_dataset_columns(
    *,
    aggregated_gdf: gpd.GeoDataFrame,
    partial_gdf: gpd.GeoDataFrame,
    geometry_name: str,
    base_columns: set[str],
    summary_path: Path,
) -> None:
    """Add dataset columns from ``partial_gdf`` into ``aggregated_gdf`` in-place."""

    for column in partial_gdf.columns:
        if column == geometry_name or column in base_columns:
            continue
        if column not in aggregated_gdf.columns:
            aggregated_gdf[column] = pd.Series(
                np.zeros(len(aggregated_gdf), dtype="int64"),
                index=aggregated_gdf.index,
            )
        left = aggregated_gdf[column].fillna(0)
        right = partial_gdf[column].fillna(0)
        left_values = left.to_numpy(dtype="int64", copy=False)
        right_values = right.to_numpy(dtype="int64", copy=False)
        if len(right_values) != len(left_values):
            raise ValueError(
                f"Mismatched row counts while merging {summary_path}: "
                f"{len(right_values)} vs {len(left_values)}"
            )
        aggregated_gdf[column] = pd.Series(
            left_values + right_values,
            index=aggregated_gdf.index,
            dtype="int64",
        )


__all__ = ["summarise_buildings_by_source", "merge_region_summaries"]
