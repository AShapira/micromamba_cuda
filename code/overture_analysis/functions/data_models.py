"""
Data models and configuration classes for the Overture Data Source Analyzer.

This module provides the core data structures used throughout the analysis system,
including configuration classes, result containers, and validation utilities.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ChartType(Enum):
    """Supported chart types for visualization"""
    BAR = "bar"
    PIE = "pie"
    STACKED_BAR = "stacked_bar"
    TIMELINE = "timeline"
    HEATMAP = "heatmap"
    SCATTER = "scatter"


class ColorScheme(Enum):
    """Supported color schemes for visualizations"""
    DEFAULT = "default"
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    CATEGORICAL = "categorical"
    PASTEL = "pastel"


class ExportFormat(Enum):
    """Supported export formats"""
    PNG = "png"
    SVG = "svg"
    HTML = "html"
    PDF = "pdf"
    GEOPARQUET = "geoparquet"
    CSV = "csv"
    GEOJSON = "geojson"


@dataclass
class FeatureTypeConfig:
    """Configuration for a specific Overture feature type"""
    theme: str
    feature_type: str
    base_path: Path
    geometry_column: str = "geometry"
    source_column: str = "sources"
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self.validate()
    
    def validate(self) -> None:
        """Validate the configuration parameters"""
        if not self.theme:
            raise ValueError("Theme cannot be empty")
        
        if not self.feature_type:
            raise ValueError("Feature type cannot be empty")
        
        if not isinstance(self.base_path, Path):
            self.base_path = Path(self.base_path)
        
        if not self.base_path.exists():
            logger.warning(f"Base path does not exist: {self.base_path}")
        
        if not self.geometry_column:
            raise ValueError("Geometry column cannot be empty")
        
        if not self.source_column:
            raise ValueError("Source column cannot be empty")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "theme": self.theme,
            "feature_type": self.feature_type,
            "base_path": str(self.base_path),
            "geometry_column": self.geometry_column,
            "source_column": self.source_column
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureTypeConfig':
        """Create instance from dictionary"""
        return cls(
            theme=data["theme"],
            feature_type=data["feature_type"],
            base_path=Path(data["base_path"]),
            geometry_column=data.get("geometry_column", "geometry"),
            source_column=data.get("source_column", "sources")
        )


@dataclass
class SourceInfo:
    """Information about a data source"""
    dataset: str
    property: Optional[str] = None
    record_id: Optional[str] = None
    confidence: Optional[float] = None
    update_time: Optional[str] = None
    
    def __post_init__(self):
        """Validate source info after initialization"""
        self.validate()
    
    def validate(self) -> None:
        """Validate the source information"""
        if not self.dataset:
            raise ValueError("Dataset cannot be empty")
        
        if self.confidence is not None:
            if not 0.0 <= self.confidence <= 1.0:
                raise ValueError("Confidence must be between 0.0 and 1.0")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "dataset": self.dataset,
            "property": self.property,
            "record_id": self.record_id,
            "confidence": self.confidence,
            "update_time": self.update_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SourceInfo':
        """Create instance from dictionary"""
        return cls(
            dataset=data["dataset"],
            property=data.get("property"),
            record_id=data.get("record_id"),
            confidence=data.get("confidence"),
            update_time=data.get("update_time")
        )


@dataclass
class FilterConfig:
    """Configuration for filtering analysis data"""
    data_sources: Optional[List[str]] = None
    min_confidence: Optional[float] = None
    max_confidence: Optional[float] = None
    metadata_filters: Optional[Dict[str, Any]] = None
    date_range: Optional[Tuple[str, str]] = None
    
    def __post_init__(self):
        """Validate filter configuration after initialization"""
        self.validate()
    
    def validate(self) -> None:
        """Validate the filter configuration"""
        if self.min_confidence is not None:
            if not 0.0 <= self.min_confidence <= 1.0:
                raise ValueError("min_confidence must be between 0.0 and 1.0")
        
        if self.max_confidence is not None:
            if not 0.0 <= self.max_confidence <= 1.0:
                raise ValueError("max_confidence must be between 0.0 and 1.0")
        
        if (self.min_confidence is not None and 
            self.max_confidence is not None and 
            self.min_confidence > self.max_confidence):
            raise ValueError("min_confidence cannot be greater than max_confidence")
    
    def matches_source(self, source_info: SourceInfo) -> bool:
        """Check if a source matches the filter criteria"""
        # Check data source filter
        if (self.data_sources is not None and 
            source_info.dataset not in self.data_sources):
            return False
        
        # Check confidence filters
        if source_info.confidence is not None:
            if (self.min_confidence is not None and 
                source_info.confidence < self.min_confidence):
                return False
            
            if (self.max_confidence is not None and 
                source_info.confidence > self.max_confidence):
                return False
        
        return True


@dataclass
class AnalysisResult:
    """Results of source analysis for a region"""
    region_id: str
    bounds: Tuple[float, float, float, float]
    feature_type: str
    total_features: int
    source_breakdown: Dict[str, int]
    coverage_metrics: Dict[str, float]
    quality_scores: Dict[str, float]
    analysis_timestamp: Optional[datetime] = field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate analysis result after initialization"""
        self.validate()
    
    def validate(self) -> None:
        """Validate the analysis result"""
        if not self.region_id:
            raise ValueError("Region ID cannot be empty")
        
        if not self.feature_type:
            raise ValueError("Feature type cannot be empty")
        
        if self.total_features < 0:
            raise ValueError("Total features cannot be negative")
        
        if len(self.bounds) != 4:
            raise ValueError("Bounds must be a tuple of 4 values (minx, miny, maxx, maxy)")
        
        minx, miny, maxx, maxy = self.bounds
        if minx >= maxx or miny >= maxy:
            raise ValueError("Invalid bounds: min values must be less than max values")
        
        # Validate source breakdown totals
        if sum(self.source_breakdown.values()) != self.total_features:
            logger.warning(
                f"Source breakdown total ({sum(self.source_breakdown.values())}) "
                f"does not match total features ({self.total_features})"
            )
    
    def get_source_percentage(self, source: str) -> float:
        """Get percentage of features from a specific source"""
        if self.total_features == 0:
            return 0.0
        return (self.source_breakdown.get(source, 0) / self.total_features) * 100
    
    def get_top_sources(self, n: int = 5) -> List[Tuple[str, int, float]]:
        """Get top N sources by feature count"""
        sorted_sources = sorted(
            self.source_breakdown.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        return [
            (source, count, self.get_source_percentage(source))
            for source, count in sorted_sources[:n]
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "region_id": self.region_id,
            "bounds": self.bounds,
            "feature_type": self.feature_type,
            "total_features": self.total_features,
            "source_breakdown": self.source_breakdown,
            "coverage_metrics": self.coverage_metrics,
            "quality_scores": self.quality_scores,
            "analysis_timestamp": self.analysis_timestamp.isoformat() if self.analysis_timestamp else None,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisResult':
        """Create instance from dictionary"""
        timestamp = None
        if data.get("analysis_timestamp"):
            timestamp = datetime.fromisoformat(data["analysis_timestamp"])
        
        return cls(
            region_id=data["region_id"],
            bounds=tuple(data["bounds"]),
            feature_type=data["feature_type"],
            total_features=data["total_features"],
            source_breakdown=data["source_breakdown"],
            coverage_metrics=data["coverage_metrics"],
            quality_scores=data["quality_scores"],
            analysis_timestamp=timestamp,
            metadata=data.get("metadata")
        )


@dataclass
class VisualizationConfig:
    """Configuration for visualization generation"""
    chart_type: Union[str, ChartType]
    color_scheme: Union[str, ColorScheme] = ColorScheme.DEFAULT
    interactive: bool = True
    export_format: Optional[Union[str, ExportFormat]] = None
    title: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    show_legend: bool = True
    custom_colors: Optional[Dict[str, str]] = None
    
    def __post_init__(self):
        """Validate and normalize configuration after initialization"""
        self.validate()
        self._normalize_enums()
    
    def validate(self) -> None:
        """Validate the visualization configuration"""
        if isinstance(self.chart_type, str):
            try:
                ChartType(self.chart_type)
            except ValueError:
                valid_types = [t.value for t in ChartType]
                raise ValueError(f"Invalid chart_type. Must be one of: {valid_types}")
        
        if isinstance(self.color_scheme, str):
            try:
                ColorScheme(self.color_scheme)
            except ValueError:
                valid_schemes = [s.value for s in ColorScheme]
                raise ValueError(f"Invalid color_scheme. Must be one of: {valid_schemes}")
        
        if self.export_format is not None and isinstance(self.export_format, str):
            try:
                ExportFormat(self.export_format)
            except ValueError:
                valid_formats = [f.value for f in ExportFormat]
                raise ValueError(f"Invalid export_format. Must be one of: {valid_formats}")
        
        if self.width is not None and self.width <= 0:
            raise ValueError("Width must be positive")
        
        if self.height is not None and self.height <= 0:
            raise ValueError("Height must be positive")
    
    def _normalize_enums(self) -> None:
        """Convert string values to enum instances"""
        if isinstance(self.chart_type, str):
            self.chart_type = ChartType(self.chart_type)
        
        if isinstance(self.color_scheme, str):
            self.color_scheme = ColorScheme(self.color_scheme)
        
        if isinstance(self.export_format, str):
            self.export_format = ExportFormat(self.export_format)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "chart_type": self.chart_type.value if isinstance(self.chart_type, ChartType) else self.chart_type,
            "color_scheme": self.color_scheme.value if isinstance(self.color_scheme, ColorScheme) else self.color_scheme,
            "interactive": self.interactive,
            "export_format": self.export_format.value if isinstance(self.export_format, ExportFormat) else self.export_format,
            "title": self.title,
            "width": self.width,
            "height": self.height,
            "show_legend": self.show_legend,
            "custom_colors": self.custom_colors
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VisualizationConfig':
        """Create instance from dictionary"""
        return cls(
            chart_type=data["chart_type"],
            color_scheme=data.get("color_scheme", ColorScheme.DEFAULT.value),
            interactive=data.get("interactive", True),
            export_format=data.get("export_format"),
            title=data.get("title"),
            width=data.get("width"),
            height=data.get("height"),
            show_legend=data.get("show_legend", True),
            custom_colors=data.get("custom_colors")
        )


@dataclass
class AnalysisConfig:
    """Complete configuration for an analysis run"""
    feature_configs: List[FeatureTypeConfig]
    bounds: Tuple[float, float, float, float]
    region_id: str
    filter_config: Optional[FilterConfig] = None
    sampling_strategy: Optional[str] = None
    aggregation_level: Optional[str] = None
    max_features: Optional[int] = None
    
    def __post_init__(self):
        """Validate analysis configuration after initialization"""
        self.validate()
    
    def validate(self) -> None:
        """Validate the complete analysis configuration"""
        if not self.feature_configs:
            raise ValueError("At least one feature configuration is required")
        
        if not self.region_id:
            raise ValueError("Region ID cannot be empty")
        
        if len(self.bounds) != 4:
            raise ValueError("Bounds must be a tuple of 4 values (minx, miny, maxx, maxy)")
        
        minx, miny, maxx, maxy = self.bounds
        if minx >= maxx or miny >= maxy:
            raise ValueError("Invalid bounds: min values must be less than max values")
        
        if self.max_features is not None and self.max_features <= 0:
            raise ValueError("max_features must be positive")
        
        # Validate all feature configurations
        for config in self.feature_configs:
            config.validate()
        
        # Validate filter configuration if provided
        if self.filter_config is not None:
            self.filter_config.validate()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "feature_configs": [config.to_dict() for config in self.feature_configs],
            "bounds": self.bounds,
            "region_id": self.region_id,
            "filter_config": self.filter_config.to_dict() if self.filter_config else None,
            "sampling_strategy": self.sampling_strategy,
            "aggregation_level": self.aggregation_level,
            "max_features": self.max_features
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisConfig':
        """Create instance from dictionary"""
        feature_configs = [
            FeatureTypeConfig.from_dict(config_data) 
            for config_data in data["feature_configs"]
        ]
        
        filter_config = None
        if data.get("filter_config"):
            filter_config = FilterConfig(**data["filter_config"])
        
        return cls(
            feature_configs=feature_configs,
            bounds=tuple(data["bounds"]),
            region_id=data["region_id"],
            filter_config=filter_config,
            sampling_strategy=data.get("sampling_strategy"),
            aggregation_level=data.get("aggregation_level"),
            max_features=data.get("max_features")
        )


def save_config_to_file(config: Union[AnalysisConfig, VisualizationConfig], 
                       filepath: Path) -> None:
    """Save configuration to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)


def load_config_from_file(filepath: Path, 
                         config_type: str) -> Union[AnalysisConfig, VisualizationConfig]:
    """Load configuration from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    if config_type == "analysis":
        return AnalysisConfig.from_dict(data)
    elif config_type == "visualization":
        return VisualizationConfig.from_dict(data)
    else:
        raise ValueError(f"Unknown config type: {config_type}")


def validate_schema(data: Dict[str, Any], schema_type: str) -> bool:
    """Validate data against expected schema"""
    try:
        if schema_type == "feature_config":
            FeatureTypeConfig.from_dict(data)
        elif schema_type == "analysis_result":
            AnalysisResult.from_dict(data)
        elif schema_type == "visualization_config":
            VisualizationConfig.from_dict(data)
        elif schema_type == "analysis_config":
            AnalysisConfig.from_dict(data)
        else:
            raise ValueError(f"Unknown schema type: {schema_type}")
        return True
    except (ValueError, KeyError, TypeError) as e:
        logger.error(f"Schema validation failed for {schema_type}: {e}")
        return False