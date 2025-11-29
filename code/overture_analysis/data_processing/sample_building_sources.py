#!/opt/conda/envs/gis/bin/python
"""
Quickly inspect Overture buildings GeoParquet files to understand source datasets.

For each GeoParquet file under the input directory the script:
- Reads only the first N rows (default 1,000) from the `sources` column.
- Counts how often each dataset appears in those rows.
- Counts combinations of datasets to see if buildings come from multiple sources.

It writes a JSON report to data/results (or a user supplied path) with per-file
and aggregate counts. No sampling randomness is used. Optionally writes sampled
rows that have multiple sources into a single GeoParquet (EPSG:4326). Optional
GeoJSON logging can capture up to 100 example buildings for every source
combination observed in the sample.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
from shapely import wkb
from shapely.geometry import mapping

LOGGER = logging.getLogger("building_sources")

DEFAULT_INPUT_DIR = Path(
    "/workspaces/micromamba_cuda/gis_data/overturemaps-us-west-2/"
    "release/2025-10-22.0/theme=buildings/type=building"
)
DEFAULT_OUTPUT_PATH = Path("/workspaces/micromamba_cuda/data/results/buildings_sources_report_100k.json")
DEFAULT_SAMPLE_SIZE = 1_000
DEFAULT_BATCH_ROWS = 50_000
DEFAULT_MULTI_OUTPUT: Path | None = None
DEFAULT_SAMPLES_GEOJSON: Path | None = None
MAX_FEATURES_PER_COMBO = 100


@dataclass(frozen=True)
class FileSummary:
    file: str
    sampled_rows: int
    total_rows_seen: int
    dataset_counts: Counter[str]
    combination_counts: Counter[str]
    rows_with_multiple_sources: int
    rows_with_no_sources: int


@dataclass(frozen=True)
class _SampleRow:
    normalized: tuple[str, ...]
    geometry: object | None
    raw_sources: object | None
    properties: dict


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _normalize_sources(value: object) -> tuple[str, ...]:
    """Return a sorted tuple of dataset names (lowercased, deduped)."""
    if value is None:
        return tuple()

    entries: Iterable[object]
    if isinstance(value, list):
        entries = value
    else:
        entries = (value,)

    datasets: set[str] = set()
    for entry in entries:
        dataset: str | None = None
        if isinstance(entry, dict):
            raw = entry.get("dataset") or entry.get("dataset_name")
            dataset = str(raw).strip() if raw is not None else None
        elif isinstance(entry, str):
            dataset = entry.strip()
        if dataset:
            datasets.add(dataset.lower())

    return tuple(sorted(datasets))


def _combo_label(combo: tuple[str, ...]) -> str:
    if not combo:
        return "none"
    return "+".join(combo)


def _sample_first_rows(
    parquet_file: pq.ParquetFile,
    sample_size: int,
    batch_rows: int,
    include_geometry: bool,
    include_properties: bool,
) -> tuple[list[_SampleRow], int]:
    """Collect the first `sample_size` rows with normalized sources (and geometry if requested)."""
    sample: list[_SampleRow] = []
    total_rows = 0
    target = sample_size
    batch_size = min(batch_rows, sample_size)
    columns = None if include_properties else (["sources"] if not include_geometry else ["geometry", "sources"])

    for batch in parquet_file.iter_batches(columns=columns, batch_size=batch_size):
        total_rows += batch.num_rows
        if include_properties:
            names = batch.schema.names
            columns_py = {name: batch.column(i).to_pylist() for i, name in enumerate(names)}
            geometries = columns_py.get("geometry", [None] * batch.num_rows)
            sources = columns_py.get("sources", [None] * batch.num_rows)
        else:
            if include_geometry:
                geometries = batch.column(0).to_pylist()
                sources = batch.column(1).to_pylist()
            else:
                geometries = [None] * batch.num_rows
                sources = batch.column(0).to_pylist()

        for row_idx, (geom_value, raw_sources) in enumerate(zip(geometries, sources)):
            normalized = _normalize_sources(raw_sources)
            properties: dict
            if include_properties:
                properties = {name: columns_py[name][row_idx] for name in names if name != "geometry"}
            else:
                properties = {}
            sample.append(
                _SampleRow(
                    normalized=normalized,
                    geometry=geom_value,
                    raw_sources=raw_sources,
                    properties=properties,
                )
            )
            if len(sample) >= target:
                return sample, total_rows

    return sample, total_rows


def _count_sample(
    sample_rows: list[_SampleRow],
    filename: str,
    total_rows_seen: int,
) -> FileSummary:
    dataset_counts: Counter[str] = Counter()
    combination_counts: Counter[str] = Counter()
    multi = 0
    none = 0

    for row in sample_rows:
        combo = row.normalized
        label = _combo_label(combo)
        combination_counts[label] += 1
        if not combo:
            none += 1
            continue
        if len(combo) > 1:
            multi += 1
        for dataset in combo:
            dataset_counts[dataset] += 1

    return FileSummary(
        file=filename,
        sampled_rows=len(sample_rows),
        total_rows_seen=total_rows_seen,
        dataset_counts=dataset_counts,
        combination_counts=combination_counts,
        rows_with_multiple_sources=multi,
        rows_with_no_sources=none,
    )


def _build_multi_batches_from_samples(sample_rows: list[_SampleRow]) -> list[pa.RecordBatch]:
    multi_rows = [row for row in sample_rows if len(row.normalized) > 1 and row.geometry is not None]
    if not multi_rows:
        return []

    geometry_array = pa.array([row.geometry for row in multi_rows], type=pa.binary())
    sources_array = pa.array([row.raw_sources for row in multi_rows])
    return [pa.record_batch({"geometry": geometry_array, "sources": sources_array})]


def _geometry_to_mapping(value: object | None) -> dict | None:
    if value is None:
        return None
    try:
        geom = wkb.loads(value)
    except Exception:  # pragma: no cover - defensive
        return None
    if geom.is_empty:
        return None
    return mapping(geom)


def _collect_combo_features(sample_rows: list[_SampleRow], file_name: str) -> list[tuple[str, dict]]:
    features: list[tuple[str, dict]] = []
    for row in sample_rows:
        if row.geometry is None:
            continue
        geom_mapping = _geometry_to_mapping(row.geometry)
        if geom_mapping is None:
            continue

        combo_label = _combo_label(row.normalized)
        props = dict(row.properties)
        props.update({"file": file_name, "combo": combo_label, "sources": row.raw_sources})
        feature = {
            "type": "Feature",
            "geometry": geom_mapping,
            "properties": props,
        }
        features.append((combo_label, feature))
    return features


def _process_file(
    path: Path,
    sample_size: int,
    batch_rows: int,
    capture_multi: bool,
    capture_samples_geojson: bool,
) -> tuple[FileSummary, list[pa.RecordBatch], list[tuple[str, dict]]]:
    multi_batches: list[pa.RecordBatch] = []
    combo_features: list[tuple[str, dict]] = []
    try:
        parquet_file = pq.ParquetFile(path)
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.error("Failed to open %s: %s", path, exc)
        empty = FileSummary(
            file=str(path),
            sampled_rows=0,
            total_rows_seen=0,
            dataset_counts=Counter(),
            combination_counts=Counter(),
            rows_with_multiple_sources=0,
            rows_with_no_sources=0,
        )
        return empty, multi_batches, combo_features

    has_geometry = "geometry" in parquet_file.schema.names
    if capture_multi and not has_geometry:
        LOGGER.warning("%s: no geometry column; skipping multi-source export for this file", path)
        capture_multi = False
    if capture_samples_geojson and not has_geometry:
        LOGGER.warning("%s: no geometry column; skipping GeoJSON sampling for this file", path)

    include_geometry = has_geometry and (capture_multi or capture_samples_geojson)
    include_properties = capture_samples_geojson
    sample_rows, total_rows = _sample_first_rows(
        parquet_file,
        sample_size,
        batch_rows,
        include_geometry=include_geometry,
        include_properties=include_properties,
    )

    summary = _count_sample(sample_rows, path.name, total_rows)

    if capture_multi and include_geometry and sample_rows:
        multi_batches.extend(_build_multi_batches_from_samples(sample_rows))

    if capture_samples_geojson and include_geometry and sample_rows:
        combo_features = _collect_combo_features(sample_rows, path.name)

    return summary, multi_batches, combo_features


def _counter_to_sorted_dict(counter: Counter[str]) -> dict[str, int]:
    return {k: counter[k] for k in sorted(counter, key=lambda x: (-counter[x], x))}


def _aggregate(summaries: list[FileSummary]) -> dict:
    aggregate_dataset_counts: Counter[str] = Counter()
    aggregate_combo_counts: Counter[str] = Counter()
    total_sampled = 0
    total_rows_seen = 0
    total_multi = 0
    total_none = 0

    per_file_reports = []
    for summary in summaries:
        total_sampled += summary.sampled_rows
        total_rows_seen += summary.total_rows_seen
        total_multi += summary.rows_with_multiple_sources
        total_none += summary.rows_with_no_sources
        aggregate_dataset_counts.update(summary.dataset_counts)
        aggregate_combo_counts.update(summary.combination_counts)

        per_file_reports.append(
            {
                "file": summary.file,
                "sampled_rows": summary.sampled_rows,
                "total_rows_seen": summary.total_rows_seen,
                "dataset_counts": _counter_to_sorted_dict(summary.dataset_counts),
                "combination_counts": _counter_to_sorted_dict(summary.combination_counts),
                "rows_with_multiple_sources": summary.rows_with_multiple_sources,
                "rows_with_no_sources": summary.rows_with_no_sources,
            }
        )

    per_file_reports.sort(key=lambda item: item["file"])

    return {
        "total_rows_seen": total_rows_seen,
        "total_sampled_rows": total_sampled,
        "rows_with_multiple_sources": total_multi,
        "rows_with_no_sources": total_none,
        "dataset_counts": _counter_to_sorted_dict(aggregate_dataset_counts),
        "combination_counts": _counter_to_sorted_dict(aggregate_combo_counts),
        "per_file": per_file_reports,
    }


def _build_geo_metadata(crs: str = "EPSG:4326") -> dict[bytes, bytes]:
    geo = {
        "version": "1.1.0",
        "primary_column": "geometry",
        "columns": {
            "geometry": {
                "encoding": "WKB",
                "geometry_types": ["Geometry"],
                "crs": crs,
            }
        },
    }
    return {b"geo": json.dumps(geo).encode("utf-8")}


def _write_multi_output(batches: list[pa.RecordBatch], output_path: Path) -> None:
    if not batches:
        LOGGER.info("No multi-source rows collected; skipping GeoParquet write.")
        return

    table = pa.Table.from_batches(batches)
    if "geometry" not in table.schema.names:
        LOGGER.warning("No geometry column found in collected rows; skipping GeoParquet write.")
        return

    schema = table.schema
    metadata = dict(schema.metadata or {})
    metadata.update(_build_geo_metadata())
    table = table.replace_schema_metadata(metadata)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path, compression="zstd")
    LOGGER.info("Wrote %s multi-source row(s) to %s", table.num_rows, output_path)


def _write_geojson_samples(combo_samples: dict[str, list[dict]], output_path: Path) -> None:
    total_features = sum(len(samples) for samples in combo_samples.values())
    if total_features == 0:
        LOGGER.info("No GeoJSON samples collected; skipping write.")
        return

    features: list[dict] = []
    for samples in combo_samples.values():
        features.extend(samples)

    feature_collection = {
        "type": "FeatureCollection",
        "features": features,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(feature_collection, f, indent=2)

    LOGGER.info(
        "Wrote %s sample feature(s) across %s combination(s) to %s",
        total_features,
        len(combo_samples),
        output_path,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing buildings GeoParquet files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Where to write the JSON report (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="How many rows to sample from each file.",
    )
    parser.add_argument(
        "--batch-rows",
        type=int,
        default=DEFAULT_BATCH_ROWS,
        help="Rows to read per batch when streaming Parquet data.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="Maximum concurrent worker threads.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit on how many files to process (for quick checks).",
    )
    parser.add_argument(
        "--multi-output",
        type=Path,
        default=DEFAULT_MULTI_OUTPUT,
        help="Deprecated: multi-source GeoParquet output is disabled; kept for compatibility.",
    )
    parser.add_argument(
        "--samples-geojson",
        type=Path,
        default=DEFAULT_SAMPLES_GEOJSON,
        help="Optional GeoJSON path; writes up to 100 sampled buildings per source combination.",
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
    configure_logging(args.log_level)

    input_dir = args.input_dir.expanduser()
    if not input_dir.exists():
        LOGGER.error("Input directory %s does not exist.", input_dir)
        return 1

    parquet_files = sorted(input_dir.rglob("*.parquet"))
    if args.max_files is not None:
        parquet_files = parquet_files[: max(0, args.max_files)]

    if not parquet_files:
        LOGGER.warning("No Parquet files found under %s", input_dir)
        return 0

    LOGGER.info(
        "Reading first %s row(s) per file from %s file(s) using %s worker(s)",
        args.sample_size,
        len(parquet_files),
        args.max_workers,
    )

    capture_multi = False
    if args.multi_output is not None:
        LOGGER.info("Multi-output write disabled; ignoring --multi-output=%s", args.multi_output)
    capture_samples_geojson = args.samples_geojson is not None
    summaries: list[FileSummary] = []
    combo_samples: dict[str, list[dict]] = defaultdict(list)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                _process_file,
                path,
                args.sample_size,
                args.batch_rows,
                capture_multi,
                capture_samples_geojson,
            ): path
            for path in parquet_files
        }
        for future in concurrent.futures.as_completed(futures):
            path = futures[future]
            try:
                summary, multi_batches, combo_features = future.result()
            except Exception as exc:  # pragma: no cover - defensive logging
                LOGGER.error("Failed processing %s: %s", path, exc)
                continue
            summaries.append(summary)
            if capture_samples_geojson and combo_features:
                for combo_label, feature in combo_features:
                    bucket = combo_samples[combo_label]
                    if len(bucket) < MAX_FEATURES_PER_COMBO:
                        bucket.append(feature)
            LOGGER.info("%s: sampled %s rows", path.name, summary.sampled_rows)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "sample_size_per_file": args.sample_size,
        "batch_rows": args.batch_rows,
        "files_processed": len(summaries),
        "aggregate": _aggregate(summaries),
    }

    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    LOGGER.info("Wrote report to %s", output_path)

    if capture_samples_geojson and args.samples_geojson is not None:
        _write_geojson_samples(combo_samples, args.samples_geojson.expanduser())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
