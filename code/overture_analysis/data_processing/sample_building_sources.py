#!/opt/conda/envs/gis/bin/python
"""
Quickly inspect Overture buildings GeoParquet files to understand source datasets.

For each GeoParquet file under the input directory the script:
- Reads only the first N rows (default 1,000) from the `sources` column.
- Counts how often each dataset appears in those rows.
- Counts combinations of datasets to see if buildings come from multiple sources.

It writes a JSON report to data/results (or a user supplied path) with per-file
and aggregate counts. No sampling randomness is used. Optionally writes sampled
rows that have multiple sources into a single GeoParquet (EPSG:4326).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

LOGGER = logging.getLogger("building_sources")

DEFAULT_INPUT_DIR = Path(
    "/workspaces/micromamba_cuda/gis_data/overturemaps-us-west-2/"
    "release/2025-10-22.0/theme=buildings/type=building"
)
DEFAULT_OUTPUT_PATH = Path("/workspaces/micromamba_cuda/data/results/buildings_sources_report_100k.json")
DEFAULT_SAMPLE_SIZE = 1_000
DEFAULT_BATCH_ROWS = 50_000
DEFAULT_MULTI_OUTPUT: Path | None = None


@dataclass(frozen=True)
class FileSummary:
    file: str
    sampled_rows: int
    total_rows_seen: int
    dataset_counts: Counter[str]
    combination_counts: Counter[str]
    rows_with_multiple_sources: int
    rows_with_no_sources: int


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


def _sample_first_sources(
    parquet_file: pq.ParquetFile,
    sample_size: int,
    batch_rows: int,
) -> tuple[list[tuple[str, ...]], int]:
    """Collect the first `sample_size` normalized sources tuples from a Parquet file."""
    sample: list[tuple[str, ...]] = []
    total_rows = 0
    target = sample_size
    batch_size = min(batch_rows, sample_size)

    for batch in parquet_file.iter_batches(columns=["sources"], batch_size=batch_size):
        values = batch.column(0).to_pylist()
        total_rows += len(values)
        for normalized in map(_normalize_sources, values):
            sample.append(normalized)
            if len(sample) >= target:
                return sample, total_rows

    return sample, total_rows


def _count_sample(
    sample: list[tuple[str, ...]],
    filename: str,
    total_rows_seen: int,
) -> FileSummary:
    dataset_counts: Counter[str] = Counter()
    combination_counts: Counter[str] = Counter()
    multi = 0
    none = 0

    for combo in sample:
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
        sampled_rows=len(sample),
        total_rows_seen=total_rows_seen,
        dataset_counts=dataset_counts,
        combination_counts=combination_counts,
        rows_with_multiple_sources=multi,
        rows_with_no_sources=none,
    )


def _process_file(
    path: Path,
    sample_size: int,
    batch_rows: int,
    capture_multi: bool,
) -> tuple[FileSummary, list[pa.RecordBatch]]:
    multi_batches: list[pa.RecordBatch] = []
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
        return empty, multi_batches

    has_geometry = "geometry" in parquet_file.schema.names
    if capture_multi and not has_geometry:
        LOGGER.warning("%s: no geometry column; skipping multi-source export for this file", path)
        capture_multi = False

    sample, total_rows = _sample_first_sources(parquet_file, sample_size, batch_rows)

    if capture_multi and sample:
        # Re-read just the rows we inspected to capture geometry + sources.
        # This keeps the fast first-N behavior while still exporting relevant rows.
        batch_size = min(batch_rows, sample_size)
        rows_remaining = len(sample)
        for batch in parquet_file.iter_batches(columns=["geometry", "sources"], batch_size=batch_size):
            values = batch.column(1).to_pylist()
            take_indices = [i for i, v in enumerate(values) if len(_normalize_sources(v)) > 1]
            if take_indices:
                multi_batches.append(batch.take(pa.array(take_indices, type=pa.int64())))
            rows_remaining -= len(values)
            if rows_remaining <= 0:
                break

    return _count_sample(sample, path.name, total_rows), multi_batches


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
        help="Optional GeoParquet path to write sampled rows that have multiple sources.",
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

    capture_multi = args.multi_output is not None
    summaries: list[FileSummary] = []
    multi_batches_all: list[pa.RecordBatch] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                _process_file,
                path,
                args.sample_size,
                args.batch_rows,
                capture_multi,
            ): path
            for path in parquet_files
        }
        for future in concurrent.futures.as_completed(futures):
            path = futures[future]
            try:
                summary, multi_batches = future.result()
            except Exception as exc:  # pragma: no cover - defensive logging
                LOGGER.error("Failed processing %s: %s", path, exc)
                continue
            summaries.append(summary)
            if capture_multi and multi_batches:
                multi_batches_all.extend(multi_batches)
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

    if capture_multi and args.multi_output is not None:
        _write_multi_output(multi_batches_all, args.multi_output.expanduser())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
