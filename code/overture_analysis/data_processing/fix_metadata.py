#!/usr/bin/env python3
"""
Fix GeoParquet metadata for summary files by setting CRS and bbox.

For every Parquet in the summary directory:
- Ensures GeoParquet metadata is present with CRS (default EPSG:4326) and geometry type Point.
- Attempts to copy bbox from the original source file (matched by prefix) when available.
- Otherwise computes bbox from the summary geometries.
- Rewrites the file in place with updated metadata.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
from shapely import from_wkb as shapely_from_wkb
from shapely import get_coordinates as shapely_get_coordinates

LOGGER = logging.getLogger("fix_metadata")
DEFAULT_BATCH_ROWS = 200_000
PREFIX_RE = re.compile(r"^(part-\d+)")


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


WKT_EPSG4326 = (
    'GEOGCRS["WGS 84",DATUM["World Geodetic System 1984",'
    'ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]]],'
    'PRIMEM["Greenwich",0,ANGLEUNIT["degree",1]],'
    'CS[ellipsoidal,2],AXIS["geodetic latitude (Lat)",north,'
    'ORDER[1],ANGLEUNIT["degree",1]],AXIS["geodetic longitude (Lon)",east,'
    'ORDER[2],ANGLEUNIT["degree",1]],ID["EPSG",4326]]'
)


def _build_geo_metadata(
    crs_value: object,
    bbox: list[float],
    crs_wkt: str | None,
    version: str = "1.0.0",
) -> dict[bytes, bytes]:
    geo = {
        "version": version,
        "primary_column": "geometry",
        "crs": crs_value,
        "columns": {
            "geometry": {
                "encoding": "WKB",
                "geometry_types": ["Point"],
                "crs": crs_value,
                "bbox": bbox,
            }
        },
        "bbox": bbox,
    }
    if crs_wkt:
        geo["wkt"] = crs_wkt
        geo["columns"]["geometry"]["wkt"] = crs_wkt
    return {b"geo": json.dumps(geo).encode("utf-8")}


def _find_original(summary_path: Path, input_dir: Path | None) -> Path | None:
    if input_dir is None:
        return None
    match = PREFIX_RE.match(summary_path.name)
    if not match:
        return None
    prefix = match.group(1)
    candidates = sorted(input_dir.rglob(f"{prefix}*.parquet"))
    return candidates[0] if candidates else None


def _crs_from_original(original_path: Path) -> tuple[object | None, str | None, str | None]:
    try:
        schema = pq.ParquetFile(original_path).schema_arrow
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.warning("Could not read original %s for CRS: %s", original_path, exc)
        return None, None, None
    meta = schema.metadata or {}
    geo_bytes = meta.get(b"geo")
    if not geo_bytes:
        return None, None, None
    try:
        geo = json.loads(geo_bytes.decode("utf-8"))
    except Exception:  # pragma: no cover - defensive
        return None, None, None
    crs_value = geo.get("crs")
    col_crs = geo.get("columns", {}).get("geometry", {}).get("crs")
    crs_wkt = geo.get("wkt") or geo.get("columns", {}).get("geometry", {}).get("wkt")
    version = geo.get("version")
    return col_crs or crs_value, crs_wkt, version


def _bbox_from_original(original_path: Path) -> list[float] | None:
    try:
        schema = pq.ParquetFile(original_path).schema_arrow
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.warning("Could not read original %s: %s", original_path, exc)
        return None
    meta = schema.metadata or {}
    geo_bytes = meta.get(b"geo")
    if not geo_bytes:
        return None
    try:
        geo = json.loads(geo_bytes.decode("utf-8"))
    except Exception:
        return None
    bbox = geo.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        return bbox
    return None


def _bbox_from_summary(summary_path: Path, batch_rows: int) -> list[float]:
    pf = pq.ParquetFile(summary_path)
    minx = miny = float("inf")
    maxx = maxy = float("-inf")

    for batch in pf.iter_batches(columns=["geometry"], batch_size=batch_rows):
        geoms = shapely_from_wkb(batch.column(0).to_numpy(zero_copy_only=False))
        if geoms.size == 0:
            continue
        coords = shapely_get_coordinates(geoms)
        xs = coords[:, 0]
        ys = coords[:, 1]
        bxmin = xs.min()
        bymin = ys.min()
        bxmax = xs.max()
        bymax = ys.max()
        minx = min(minx, float(bxmin))
        miny = min(miny, float(bymin))
        maxx = max(maxx, float(bxmax))
        maxy = max(maxy, float(bymax))

    if minx == float("inf"):
        return [0.0, 0.0, 0.0, 0.0]
    return [float(minx), float(miny), float(maxx), float(maxy)]


def _rewrite_with_metadata(parquet_path: Path, geo_metadata: dict[bytes, bytes]) -> None:
    pf = pq.ParquetFile(parquet_path)
    base_schema = pf.schema_arrow

    # Add GeoArrow extension metadata on geometry field for GDAL/QGIS compatibility.
    fields = list(base_schema)
    geom_idx = base_schema.get_field_index("geometry")
    if geom_idx != -1:
        geom_field = fields[geom_idx]
        crs_meta_value = json.loads(geo_metadata[b"geo"].decode()).get("crs", "EPSG:4326")
        ext_meta = {
            b"ARROW:extension:name": b"geoarrow.wkb",
            b"ARROW:extension:metadata": json.dumps(
                {
                    "crs": crs_meta_value,
                    "encoding": "WKB",
                    "geometry_type": ["Point"],
                    "edges": "planar",
                }
            ).encode(),
        }
        merged_meta = {**(geom_field.metadata or {}), **ext_meta}
        fields[geom_idx] = geom_field.with_metadata(merged_meta)
    schema = pa.schema(fields).with_metadata({**(base_schema.metadata or {}), **geo_metadata})

    with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet", dir=parquet_path.parent) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with pq.ParquetWriter(tmp_path, schema, compression="zstd") as writer:
            for rg in range(pf.num_row_groups):
                table = pf.read_row_group(rg)
                writer.write_table(table)
        try:
            tmp_path.replace(parquet_path)
        except PermissionError:
            parquet_path.unlink(missing_ok=True)
            tmp_path.replace(parquet_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _process_file(summary_path: Path, input_dir: Path | None, crs: str, batch_rows: int) -> tuple[str, str]:
    original_path = _find_original(summary_path, input_dir)
    bbox = _bbox_from_original(original_path) if original_path else None
    if bbox is None:
        bbox = _bbox_from_summary(summary_path, batch_rows)

    orig_crs_value, orig_crs_wkt, orig_version = _crs_from_original(original_path) if original_path else (None, None, None)

    if orig_crs_value is None and str(crs).upper().startswith("EPSG:"):
        try:
            epsg_code = int(str(crs).split(":")[1])
        except Exception:
            epsg_code = 4326
        crs_value = {
            "type": "GeographicCRS",
            "name": "WGS 84" if epsg_code == 4326 else crs,
            "id": {"authority": "EPSG", "code": epsg_code},
        }
        crs_wkt = WKT_EPSG4326 if epsg_code == 4326 else None
    else:
        crs_value = orig_crs_value or crs
        crs_wkt = orig_crs_wkt if orig_crs_wkt else (WKT_EPSG4326 if str(crs).upper() == "EPSG:4326" else None)

    version = orig_version or "1.0.0"

    geo_meta = _build_geo_metadata(crs_value, bbox, crs_wkt, version)
    _rewrite_with_metadata(summary_path, geo_meta)
    source_note = "original bbox" if original_path and bbox else "computed bbox"
    return summary_path.name, f"updated ({source_note})"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-dir",
        type=Path,
        required=True,
        help="Directory containing summary GeoParquet files to fix.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Optional original buildings directory to copy bbox from (matched by prefix).",
    )
    parser.add_argument(
        "--crs",
        default="EPSG:4326",
        help="CRS to set in GeoParquet metadata (default: EPSG:4326).",
    )
    parser.add_argument(
        "--batch-rows",
        type=int,
        default=DEFAULT_BATCH_ROWS,
        help="Rows per batch when computing bbox from summary geometries.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="Parallel worker count.",
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
    summary_dir = args.summary_dir.expanduser()
    input_dir = args.input_dir.expanduser() if args.input_dir else None

    if not summary_dir.exists():
        LOGGER.error("Summary dir %s does not exist", summary_dir)
        return 1

    summary_files = sorted(summary_dir.rglob("*.parquet"))
    if not summary_files:
        LOGGER.warning("No Parquet files found under %s", summary_dir)
        return 0

    LOGGER.info("Fixing metadata for %s file(s)", len(summary_files))
    results: list[tuple[str, str]] = []

    if args.max_workers == 1:
        for path in summary_files:
            results.append(_process_file(path, input_dir, args.crs, args.batch_rows))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futs = {
                executor.submit(_process_file, path, input_dir, args.crs, args.batch_rows): path
                for path in summary_files
            }
            for fut in concurrent.futures.as_completed(futs):
                results.append(fut.result())

    for fname, msg in results:
        LOGGER.info("%s: %s", fname, msg)
    LOGGER.info("Completed metadata fixes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
