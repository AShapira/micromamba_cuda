"""Utilities for repeated bounding-box retrievals from Overture buildings GeoParquet files.

Strategy overview
-----------------
The Overture buildings release for ``theme=buildings/type=building`` consists of a few
hundred GeoParquet partitions stored on local SSD.  Reading every file for each query is
wasteful, especially when a laptop-class machine (32 GB RAM, i7 CPU) needs to answer
thousands of requests for different bounding boxes.

To keep queries responsive:

* Inspect the GeoParquet metadata once to build a lightweight in-memory index that maps
  each file to its spatial bounding box (``bbox``) and geometry column.
* Use that index to shortlist only the files whose bounding boxes intersect the query
  envelope.  The shortlist step avoids touching the majority of files for most queries.
* Maintain a persistent DuckDB in-memory connection with the Spatial extension loaded so
  that subsequent queries can scan just the shortlisted files and apply
  ``ST_Intersects`` against the geometry column.  DuckDB reads Parquet lazily, so only
  the required row groups are decompressed.
* Reuse the same connection for every call to minimise overhead.  The index stays small
  (hundreds of entries) so it easily fits in memory.

This module exposes :class:`BuildingRetriever`, which encapsulates the above strategy and
can be reused safely across many bounding-box lookups.
"""

from __future__ import annotations

import json
import math
import time
import ast
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import duckdb
from pyarrow import parquet as pq


BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class FileIndexEntry:
    """Holds bounding-box metadata for a single GeoParquet file."""

    path: Path
    bbox: BBox

    def intersects(self, query_bbox: BBox) -> bool:
        """Return ``True`` when this file's bounding box intersects ``query_bbox``."""

        minx, miny, maxx, maxy = self.bbox
        qminx, qminy, qmaxx, qmaxy = query_bbox
        if maxx < qminx or qmaxx < minx:
            return False
        if maxy < qminy or qmaxy < miny:
            return False
        return True


class BuildingRetriever:
    """Retrieve Overture buildings records that intersect a bounding box.

    Parameters
    ----------
    base_path:
        Directory containing the ``theme=buildings/type=building`` GeoParquet files.
    auto_install_spatial:
        Install the DuckDB ``spatial`` extension when it is missing.  Disable when the
        environment forbids extension installs.
    load_spatial:
        Load the ``spatial`` extension.  Set to ``False`` when you know the geometry
        column is already exploded into longitude/latitude columns and you plan to apply
        custom filtering.  The default requires the extension because GeoParquet stores
        geometries as WKB.
    """

    def __init__(
        self,
        base_path: Path | str,
        *,
        auto_install_spatial: bool = True,
        load_spatial: bool = True,
    ) -> None:
        self.base_path = Path(base_path)
        if not self.base_path.exists():
            raise FileNotFoundError(f"Base path does not exist: {self.base_path}")

        self._file_index: List[FileIndexEntry] = []
        self._geometry_column: Optional[str] = None
        self._build_index()

        self._conn: Optional[duckdb.DuckDBPyConnection] = duckdb.connect(
            database=":memory:", read_only=False
        )
        self._spatial_loaded = False
        if load_spatial:
            self._ensure_spatial_extension(auto_install_spatial)

    @property
    def geometry_column(self) -> str:
        if not self._geometry_column:
            raise RuntimeError("Geometry column not detected while indexing GeoParquet files.")
        return self._geometry_column

    @property
    def file_count(self) -> int:
        return len(self._file_index)

    def _build_index(self) -> None:
        geo_column: Optional[str] = None
        for file_path in sorted(self.base_path.rglob("*.parquet")):
            metadata = pq.ParquetFile(file_path).metadata.metadata or {}
            decoded = {
                (k.decode("utf-8") if isinstance(k, (bytes, bytearray)) else k):
                (v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else v)
                for k, v in metadata.items()
            }
            geo_payload = decoded.get("geo")
            if not geo_payload:
                continue
            try:
                geo_json = json.loads(geo_payload)
            except json.JSONDecodeError as exc:  # pragma: no cover - metadata is trusted
                raise ValueError(f"Malformed GeoParquet metadata in {file_path}") from exc

            if geo_column is None:
                geo_column = geo_json.get("primary_column")
                if not geo_column:
                    raise ValueError(f"Primary geometry column missing in {file_path}")
            elif geo_column != geo_json.get("primary_column"):
                raise ValueError(
                    "Detected multiple geometry columns across files."
                    " This retriever expects a consistent schema."
                )

            columns_meta = geo_json.get("columns", {})
            bbox = columns_meta.get(geo_column, {}).get("bbox") if columns_meta else None
            if not bbox or len(bbox) != 4:
                raise ValueError(f"Bounding box metadata missing in {file_path}")

            minx, miny, maxx, maxy = map(float, bbox)
            self._file_index.append(
                FileIndexEntry(path=file_path, bbox=(minx, miny, maxx, maxy))
            )

        if not self._file_index:
            raise FileNotFoundError(
                f"No GeoParquet files with Geo metadata found under {self.base_path}"
            )
        self._geometry_column = geo_column

    def _ensure_spatial_extension(self, auto_install: bool) -> None:
        conn = self._connection()
        try:
            conn.execute("LOAD spatial;")
            self._spatial_loaded = True
            return
        except duckdb.IOException:
            pass
        if not auto_install:
            raise RuntimeError("DuckDB spatial extension is required but could not be loaded.")
        conn.execute("INSTALL spatial;")
        conn.execute("LOAD spatial;")
        self._spatial_loaded = True

    def _candidate_files(self, query_bbox: BBox) -> List[Path]:
        return [entry.path for entry in self._file_index if entry.intersects(query_bbox)]

    def _connection(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            raise RuntimeError("Retriever has been closed.")
        return self._conn

    def query_bbox(
        self,
        minx: float,
        miny: float,
        maxx: float,
        maxy: float,
        *,
        columns: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ):
        """Return records whose geometry intersects the bounding box.

        Parameters
        ----------
        minx, miny, maxx, maxy:
            Bounding box coordinates (WGS84 lon/lat) with ``min`` <= ``max``.
        columns:
            Optional subset of columns to project before returning.
        limit:
            Optional cap on the number of rows returned (applied after spatial filtering).

        Returns
        -------
        pandas.DataFrame
            Rows intersecting the supplied bounding box.  The DataFrame is empty when no
            intersections are found.
        """

        if minx > maxx or miny > maxy:
            raise ValueError("Bounding box minimums must be less than maximums.")
        bbox = (float(minx), float(miny), float(maxx), float(maxy))
        candidate_files = self._candidate_files(bbox)
        if not candidate_files:
            import pandas as pd  # Local import to keep pandas optional until needed

            return pd.DataFrame(columns=list(columns) if columns else None)

        if not self._spatial_loaded:
            raise RuntimeError(
                "Spatial extension is not available; cannot apply geometry intersection filtering."
            )

        conn = self._connection()
        relation = conn.read_parquet([str(path) for path in candidate_files])

        geom_col = self.geometry_column
        if columns:
            if geom_col not in columns:
                projection = list(dict.fromkeys([geom_col, *columns]))
            else:
                projection = list(dict.fromkeys(columns))
            relation = relation.project(", ".join(projection))
        envelope = f"ST_MakeEnvelope({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})"
        relation = relation.filter(f"ST_Intersects({geom_col}, {envelope})")

        if limit is not None:
            relation = relation.limit(limit)

        result = relation.to_df()
        if columns and geom_col not in columns:
            result = result.drop(columns=[geom_col], errors="ignore")
        return result

    def iter_bbox(
        self,
        minx: float,
        miny: float,
        maxx: float,
        maxy: float,
        *,
        chunk_size: int = 10000,
        columns: Optional[Sequence[str]] = None,
    ) -> Iterable:
        """Stream rows intersecting the bounding box in chunks.

        Useful when the result set is large and you do not want to materialise it all at
        once.  Each yielded chunk is a pandas DataFrame.
        """

        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")

        bbox = (float(minx), float(miny), float(maxx), float(maxy))
        candidate_files = self._candidate_files(bbox)
        if not candidate_files:
            return

        if not self._spatial_loaded:
            raise RuntimeError(
                "Spatial extension is not available; cannot apply geometry intersection filtering."
            )

        conn = self._connection()
        relation = conn.read_parquet([str(path) for path in candidate_files])
        geom_col = self.geometry_column
        if columns:
            if geom_col not in columns:
                projection = list(dict.fromkeys([geom_col, *columns]))
            else:
                projection = list(dict.fromkeys(columns))
            relation = relation.project(", ".join(projection))
        envelope = f"ST_MakeEnvelope({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})"
        relation = relation.filter(f"ST_Intersects({geom_col}, {envelope})")

        offset = 0
        while True:
            chunk_relation = relation.limit(chunk_size, offset=offset)
            df = chunk_relation.to_df()
            if df.empty:
                break
            if columns and geom_col not in columns:
                df = df.drop(columns=[geom_col], errors="ignore")
            yield df
            offset += chunk_size

    def collect_bbox_statistics(
        self,
        rows,
        *,
        geometry_field: str = "geometry",
        metadata_fields: Optional[Sequence[str]] = None,
        include_osm_breakdown: bool = True,
    ) -> Tuple[List[dict], List[dict]]:
        """Summarise bounding-box queries for an iterable of rows.

        Parameters
        ----------
        rows:
            A pandas (Geo)DataFrame, Series iterator, or iterable of mapping/objects that
            provide a geometry and optional metadata fields.
        geometry_field:
            Field name used to access the geometry from each row.
        metadata_fields:
            Additional field names to copy into the results records.
        include_osm_breakdown:
            When ``True`` (default) counts how many retrieved buildings list
            ``dataset == 'OpenStreetMap'`` in their ``sources`` arrays.

        Returns
        -------
        tuple[list[dict], list[dict]]
            Two lists of dictionaries: one suitable for GeoDataFrame construction and a
            second containing per-query log metadata.
        """

        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - pandas expected in notebooks
            raise RuntimeError("pandas is required for collect_bbox_statistics") from exc

        metadata_fields = tuple(metadata_fields or ())

        def _iter_rows() -> Iterator[tuple[object, object]]:
            if hasattr(rows, "iterrows"):
                yield from rows.iterrows()
            else:
                for idx, row in enumerate(rows):
                    yield idx, row

        def _get_value(record, field):
            if isinstance(record, dict):
                return record.get(field)
            if hasattr(record, "get"):
                try:
                    return record.get(field)
                except TypeError:
                    pass
            if hasattr(record, field):
                return getattr(record, field)
            return None

        def _normalise_index(idx: object) -> object:
            try:
                return int(idx)  # handles numpy integer types
            except (TypeError, ValueError):
                return idx

        def _has_osm_source(entries) -> bool:
            if entries is None:
                return False
            if 'pd' in globals() and entries is getattr(pd, 'NA', object()):
                return False
            if isinstance(entries, float) and math.isnan(entries):
                return False

            # Normalise various container shapes that DuckDB may return.
            if isinstance(entries, Mapping):
                iterable = [entries]
            elif isinstance(entries, Sequence) and not isinstance(entries, (str, bytes, bytearray)):
                iterable = list(entries)
            else:
                if hasattr(entries, "items"):
                    try:
                        iterable = [dict(entries)]
                    except Exception:  # pragma: no cover - best-effort fallback
                        return False
                elif hasattr(entries, "to_pylist"):
                    iterable = list(entries.to_pylist())
                else:
                    try:
                        iterable = list(entries)
                    except TypeError:
                        return False

            for item in iterable:
                if item is None:
                    continue
                if not isinstance(item, Mapping) and hasattr(item, "items"):
                    try:
                        item = dict(item)
                    except Exception:
                        pass
                if isinstance(item, Mapping):
                    dataset = item.get("dataset")
                elif hasattr(item, "get"):
                    dataset = item.get("dataset")
                else:
                    dataset = getattr(item, "dataset", None)
                if dataset == "OpenStreetMap":
                    return True
            return False

        results_records: List[dict] = []
        log_entries: List[dict] = []

        query_columns = ["sources"] if include_osm_breakdown else None

        for idx, row in _iter_rows():
            geom = _get_value(row, geometry_field)
            if geom is None:
                continue
            if hasattr(geom, "is_empty") and geom.is_empty:
                continue
            try:
                minx, miny, maxx, maxy = geom.bounds
            except AttributeError as exc:
                raise ValueError("Geometry objects must expose a 'bounds' tuple.") from exc

            bbox = (float(minx), float(miny), float(maxx), float(maxy))
            candidate_files = self._candidate_files(bbox)

            start = time.perf_counter()
            buildings_df = self.query_bbox(*bbox, columns=query_columns)
            duration = time.perf_counter() - start

            count = int(len(buildings_df))
            osm_count = 0
            if include_osm_breakdown and not buildings_df.empty and "sources" in buildings_df.columns:
                osm_count = int(buildings_df["sources"].apply(_has_osm_source).sum())

            record = {field: _get_value(row, field) for field in metadata_fields}
            record.setdefault("geometry", geom)
            record["buildings_count"] = count
            if include_osm_breakdown:
                record["osm_buildings_count"] = osm_count
            results_records.append(record)

            log_entry = {
                "feature_index": _normalise_index(idx),
                "minx": minx,
                "miny": miny,
                "maxx": maxx,
                "maxy": maxy,
                "buildings_count": count,
                "candidate_file_count": len(candidate_files),
                "duration_seconds": duration,
            }
            if include_osm_breakdown:
                log_entry["osm_buildings_count"] = osm_count
            log_entries.append(log_entry)

        return results_records, log_entries

    def close(self) -> None:
        """Close the underlying DuckDB connection."""

        if self._conn is not None:
            self._conn.close()
            self._conn = None


__all__ = ["BuildingRetriever", "FileIndexEntry", "BBox"]
