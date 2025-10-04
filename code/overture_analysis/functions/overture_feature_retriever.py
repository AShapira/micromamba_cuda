"""Universal feature retriever for any Overture Maps feature type.

This module extends the BuildingRetriever pattern to support all Overture Maps themes and
feature types, providing a unified interface for spatial queries across different data types.

Key Features:
- Theme and type discovery from directory structure
- Consistent spatial querying API across all feature types
- Schema introspection for dynamic field discovery
- Multi-feature type management and comparative analysis
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import duckdb
from pyarrow import parquet as pq

from building_retriever import BBox, FileIndexEntry


@dataclass(frozen=True)
class FeatureTypeInfo:
    """Information about an available Overture feature type."""
    theme: str
    feature_type: str
    path: Path
    file_count: int
    geometry_column: Optional[str] = None
    schema_info: Optional[Dict[str, Any]] = None


class OvertureFeatureRetriever:
    """Universal retriever for any Overture Maps feature type.
    
    Extends the BuildingRetriever pattern to support all Overture themes and types.
    
    Parameters
    ----------
    base_path:
        Directory containing Overture GeoParquet files organized by theme/type
    theme:
        Overture theme (e.g., 'buildings', 'places', 'transportation', 'divisions')
    feature_type:
        Feature type within the theme (e.g., 'building', 'place', 'segment')
    auto_install_spatial:
        Install the DuckDB spatial extension when missing
    load_spatial:
        Load the spatial extension for geometry operations
    """
    
    def __init__(
        self,
        base_path: Path | str,
        theme: str,
        feature_type: str,
        *,
        auto_install_spatial: bool = True,
        load_spatial: bool = True,
    ) -> None:
        self.base_path = Path(base_path)
        self.theme = theme
        self.feature_type = feature_type
        
        # Construct the path to the specific feature type
        self.feature_path = self.base_path / f"theme={theme}" / f"type={feature_type}"
        
        if not self.feature_path.exists():
            raise FileNotFoundError(
                f"Feature type path does not exist: {self.feature_path}"
            )
        
        self._file_index: List[FileIndexEntry] = []
        self._geometry_column: Optional[str] = None
        self._schema_info: Dict[str, Any] = {}
        self._build_index()
        
        # Initialize DuckDB connection
        self._conn: Optional[duckdb.DuckDBPyConnection] = duckdb.connect(
            database=":memory:", read_only=False
        )
        self._spatial_loaded = False
        if load_spatial:
            self._ensure_spatial_extension(auto_install_spatial)
    
    @property
    def geometry_column(self) -> str:
        """Get the geometry column name for this feature type."""
        if not self._geometry_column:
            raise RuntimeError(
                f"Geometry column not detected for {self.theme}/{self.feature_type}"
            )
        return self._geometry_column
    
    @property
    def file_count(self) -> int:
        """Get the number of indexed files."""
        return len(self._file_index)
    
    @property
    def schema_info(self) -> Dict[str, Any]:
        """Get schema information for this feature type."""
        return self._schema_info.copy()
    
    def _build_index(self) -> None:
        """Build spatial index from GeoParquet metadata."""
        geo_column: Optional[str] = None
        schema_columns: Dict[str, str] = {}
        
        parquet_files = list(self.feature_path.rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(
                f"No parquet files found in {self.feature_path}"
            )
        
        for file_path in sorted(parquet_files):
            try:
                pf = pq.ParquetFile(file_path)
                metadata = pf.metadata.metadata or {}
                
                # Decode metadata keys and values
                decoded = {
                    (k.decode("utf-8") if isinstance(k, (bytes, bytearray)) else k):
                    (v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else v)
                    for k, v in metadata.items()
                }
                
                # Extract GeoParquet metadata
                geo_payload = decoded.get("geo")
                if not geo_payload:
                    continue
                
                try:
                    geo_json = json.loads(geo_payload)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed GeoParquet metadata in {file_path}") from exc
                
                # Get geometry column
                if geo_column is None:
                    geo_column = geo_json.get("primary_column")
                    if not geo_column:
                        raise ValueError(f"Primary geometry column missing in {file_path}")
                elif geo_column != geo_json.get("primary_column"):
                    raise ValueError(
                        "Detected multiple geometry columns across files. "
                        "This retriever expects a consistent schema."
                    )
                
                # Extract bounding box
                columns_meta = geo_json.get("columns", {})
                bbox = columns_meta.get(geo_column, {}).get("bbox") if columns_meta else None
                if not bbox or len(bbox) != 4:
                    raise ValueError(f"Bounding box metadata missing in {file_path}")
                
                minx, miny, maxx, maxy = map(float, bbox)
                self._file_index.append(
                    FileIndexEntry(path=file_path, bbox=(minx, miny, maxx, maxy))
                )
                
                # Collect schema information from first file
                if not schema_columns:
                    schema = pf.schema_arrow
                    for field in schema:
                        schema_columns[field.name] = str(field.type)
                
            except Exception as e:
                # Log warning but continue with other files
                print(f"Warning: Could not process {file_path}: {e}")
                continue
        
        if not self._file_index:
            raise FileNotFoundError(
                f"No valid GeoParquet files found in {self.feature_path}"
            )
        
        self._geometry_column = geo_column
        self._schema_info = {
            "geometry_column": geo_column,
            "columns": schema_columns,
            "file_count": len(self._file_index),
            "theme": self.theme,
            "feature_type": self.feature_type
        }
    
    def _ensure_spatial_extension(self, auto_install: bool) -> None:
        """Ensure DuckDB spatial extension is loaded."""
        conn = self._connection()
        try:
            conn.execute("LOAD spatial;")
            self._spatial_loaded = True
            return
        except duckdb.IOException:
            pass
        
        if not auto_install:
            raise RuntimeError(
                "DuckDB spatial extension is required but could not be loaded."
            )
        
        conn.execute("INSTALL spatial;")
        conn.execute("LOAD spatial;")
        self._spatial_loaded = True
    
    def _candidate_files(self, query_bbox: BBox) -> List[Path]:
        """Get files whose bounding boxes intersect the query bbox."""
        return [entry.path for entry in self._file_index if entry.intersects(query_bbox)]
    
    def _connection(self) -> duckdb.DuckDBPyConnection:
        """Get the DuckDB connection."""
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
        **kwargs
    ):
        """Query features that intersect the bounding box.
        
        Parameters
        ----------
        minx, miny, maxx, maxy:
            Bounding box coordinates (WGS84 lon/lat)
        columns:
            Optional subset of columns to return
        limit:
            Optional limit on number of rows returned
        **kwargs:
            Additional query parameters (for future extensions)
        
        Returns
        -------
        pandas.DataFrame
            Features intersecting the bounding box
        """
        if minx > maxx or miny > maxy:
            raise ValueError("Bounding box minimums must be less than maximums.")
        
        bbox = (float(minx), float(miny), float(maxx), float(maxy))
        candidate_files = self._candidate_files(bbox)
        
        if not candidate_files:
            import pandas as pd
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
        
        # Apply spatial filter
        envelope = f"ST_MakeEnvelope({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})"
        relation = relation.filter(f"ST_Intersects({geom_col}, {envelope})")
        
        if limit is not None:
            relation = relation.limit(limit)
        
        result = relation.to_df()
        if columns and geom_col not in columns:
            result = result.drop(columns=[geom_col], errors="ignore")
        
        return result
    
    @classmethod
    def get_available_themes(cls, base_path: Path | str) -> List[str]:
        """Discover available Overture themes in the base directory.
        
        Parameters
        ----------
        base_path:
            Base directory containing Overture data
            
        Returns
        -------
        List[str]
            Available theme names
        """
        base_path = Path(base_path)
        if not base_path.exists():
            return []
        
        themes = []
        for item in base_path.iterdir():
            if item.is_dir() and item.name.startswith("theme="):
                theme_name = item.name.replace("theme=", "")
                themes.append(theme_name)
        
        return sorted(themes)
    
    @classmethod
    def get_available_types(cls, base_path: Path | str, theme: str) -> List[str]:
        """Discover available feature types for a theme.
        
        Parameters
        ----------
        base_path:
            Base directory containing Overture data
        theme:
            Theme name to inspect
            
        Returns
        -------
        List[str]
            Available feature type names for the theme
        """
        base_path = Path(base_path)
        theme_path = base_path / f"theme={theme}"
        
        if not theme_path.exists():
            return []
        
        types = []
        for item in theme_path.iterdir():
            if item.is_dir() and item.name.startswith("type="):
                type_name = item.name.replace("type=", "")
                types.append(type_name)
        
        return sorted(types)
    
    def close(self) -> None:
        """Close the underlying DuckDB connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None


class MultiFeatureRetriever:
    """Manages multiple feature retrievers for comparative analysis.
    
    Allows querying multiple Overture feature types simultaneously and
    provides unified access to results across different themes/types.
    
    Parameters
    ----------
    base_path:
        Base directory containing Overture data organized by theme/type
    auto_install_spatial:
        Install DuckDB spatial extension when missing
    load_spatial:
        Load spatial extension for geometry operations
    """
    
    def __init__(
        self,
        base_path: Path | str,
        *,
        auto_install_spatial: bool = True,
        load_spatial: bool = True,
    ) -> None:
        self.base_path = Path(base_path)
        self._retrievers: Dict[Tuple[str, str], OvertureFeatureRetriever] = {}
        self._auto_install_spatial = auto_install_spatial
        self._load_spatial = load_spatial
    
    def add_feature_type(self, theme: str, feature_type: str) -> None:
        """Add a feature type to the multi-retriever.
        
        Parameters
        ----------
        theme:
            Overture theme name
        feature_type:
            Feature type name within the theme
        """
        key = (theme, feature_type)
        if key in self._retrievers:
            return  # Already added
        
        try:
            retriever = OvertureFeatureRetriever(
                self.base_path,
                theme,
                feature_type,
                auto_install_spatial=self._auto_install_spatial,
                load_spatial=self._load_spatial,
            )
            self._retrievers[key] = retriever
        except FileNotFoundError as e:
            print(f"Warning: Could not add {theme}/{feature_type}: {e}")
    
    def remove_feature_type(self, theme: str, feature_type: str) -> None:
        """Remove a feature type from the multi-retriever.
        
        Parameters
        ----------
        theme:
            Overture theme name
        feature_type:
            Feature type name within the theme
        """
        key = (theme, feature_type)
        if key in self._retrievers:
            self._retrievers[key].close()
            del self._retrievers[key]
    
    def query_all_types(
        self,
        minx: float,
        miny: float,
        maxx: float,
        maxy: float,
        *,
        columns: Optional[Dict[Tuple[str, str], Sequence[str]]] = None,
        limit: Optional[int] = None,
        **kwargs
    ) -> Dict[Tuple[str, str], Any]:
        """Query all registered feature types for the bounding box.
        
        Parameters
        ----------
        minx, miny, maxx, maxy:
            Bounding box coordinates (WGS84 lon/lat)
        columns:
            Optional mapping of (theme, type) -> column list for each feature type
        limit:
            Optional limit on rows returned per feature type
        **kwargs:
            Additional query parameters
            
        Returns
        -------
        Dict[Tuple[str, str], pandas.DataFrame]
            Results keyed by (theme, feature_type) tuples
        """
        results = {}
        
        for key, retriever in self._retrievers.items():
            try:
                feature_columns = columns.get(key) if columns else None
                result = retriever.query_bbox(
                    minx, miny, maxx, maxy,
                    columns=feature_columns,
                    limit=limit,
                    **kwargs
                )
                results[key] = result
            except Exception as e:
                print(f"Warning: Query failed for {key[0]}/{key[1]}: {e}")
                import pandas as pd
                results[key] = pd.DataFrame()
        
        return results
    
    def get_feature_types(self) -> List[Tuple[str, str]]:
        """Get list of registered feature types.
        
        Returns
        -------
        List[Tuple[str, str]]
            List of (theme, feature_type) tuples
        """
        return list(self._retrievers.keys())
    
    def get_schema_info(self, theme: str, feature_type: str) -> Dict[str, Any]:
        """Get schema information for a specific feature type.
        
        Parameters
        ----------
        theme:
            Overture theme name
        feature_type:
            Feature type name
            
        Returns
        -------
        Dict[str, Any]
            Schema information for the feature type
        """
        key = (theme, feature_type)
        if key not in self._retrievers:
            raise ValueError(f"Feature type {theme}/{feature_type} not registered")
        
        return self._retrievers[key].schema_info
    
    def get_all_schema_info(self) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """Get schema information for all registered feature types.
        
        Returns
        -------
        Dict[Tuple[str, str], Dict[str, Any]]
            Schema information keyed by (theme, feature_type)
        """
        return {
            key: retriever.schema_info
            for key, retriever in self._retrievers.items()
        }
    
    def close(self) -> None:
        """Close all underlying retrievers."""
        for retriever in self._retrievers.values():
            retriever.close()
        self._retrievers.clear()


__all__ = [
    "OvertureFeatureRetriever",
    "MultiFeatureRetriever", 
    "FeatureTypeInfo"
]