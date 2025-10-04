"""
Tests for data models and configuration classes.
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from data_models import (
    FeatureTypeConfig,
    SourceInfo,
    FilterConfig,
    AnalysisResult,
    VisualizationConfig,
    AnalysisConfig,
    ChartType,
    ColorScheme,
    ExportFormat,
    save_config_to_file,
    load_config_from_file,
    validate_schema
)


class TestFeatureTypeConfig:
    """Tests for FeatureTypeConfig class"""
    
    def test_valid_config(self):
        """Test creating a valid feature type configuration"""
        config = FeatureTypeConfig(
            theme="buildings",
            feature_type="building",
            base_path=Path("/tmp/test")
        )
        
        assert config.theme == "buildings"
        assert config.feature_type == "building"
        assert config.base_path == Path("/tmp/test")
        assert config.geometry_column == "geometry"
        assert config.source_column == "sources"
    
    def test_custom_columns(self):
        """Test configuration with custom column names"""
        config = FeatureTypeConfig(
            theme="places",
            feature_type="place",
            base_path=Path("/tmp/test"),
            geometry_column="geom",
            source_column="data_sources"
        )
        
        assert config.geometry_column == "geom"
        assert config.source_column == "data_sources"
    
    def test_empty_theme_validation(self):
        """Test validation fails for empty theme"""
        with pytest.raises(ValueError, match="Theme cannot be empty"):
            FeatureTypeConfig(
                theme="",
                feature_type="building",
                base_path=Path("/tmp/test")
            )
    
    def test_empty_feature_type_validation(self):
        """Test validation fails for empty feature type"""
        with pytest.raises(ValueError, match="Feature type cannot be empty"):
            FeatureTypeConfig(
                theme="buildings",
                feature_type="",
                base_path=Path("/tmp/test")
            )
    
    def test_empty_geometry_column_validation(self):
        """Test validation fails for empty geometry column"""
        with pytest.raises(ValueError, match="Geometry column cannot be empty"):
            FeatureTypeConfig(
                theme="buildings",
                feature_type="building",
                base_path=Path("/tmp/test"),
                geometry_column=""
            )
    
    def test_path_conversion(self):
        """Test automatic path conversion from string"""
        config = FeatureTypeConfig(
            theme="buildings",
            feature_type="building",
            base_path="/tmp/test"  # String path
        )
        
        assert isinstance(config.base_path, Path)
        assert config.base_path == Path("/tmp/test")
    
    def test_to_dict(self):
        """Test conversion to dictionary"""
        config = FeatureTypeConfig(
            theme="buildings",
            feature_type="building",
            base_path=Path("/tmp/test")
        )
        
        result = config.to_dict()
        expected = {
            "theme": "buildings",
            "feature_type": "building",
            "base_path": "/tmp/test",
            "geometry_column": "geometry",
            "source_column": "sources"
        }
        
        assert result == expected
    
    def test_from_dict(self):
        """Test creation from dictionary"""
        data = {
            "theme": "buildings",
            "feature_type": "building",
            "base_path": "/tmp/test",
            "geometry_column": "geom",
            "source_column": "data_sources"
        }
        
        config = FeatureTypeConfig.from_dict(data)
        
        assert config.theme == "buildings"
        assert config.feature_type == "building"
        assert config.base_path == Path("/tmp/test")
        assert config.geometry_column == "geom"
        assert config.source_column == "data_sources"


class TestSourceInfo:
    """Tests for SourceInfo class"""
    
    def test_minimal_source_info(self):
        """Test creating source info with minimal data"""
        source = SourceInfo(dataset="osm")
        
        assert source.dataset == "osm"
        assert source.property is None
        assert source.record_id is None
        assert source.confidence is None
        assert source.update_time is None
    
    def test_complete_source_info(self):
        """Test creating source info with all fields"""
        source = SourceInfo(
            dataset="osm",
            property="building",
            record_id="123456",
            confidence=0.95,
            update_time="2023-01-01T00:00:00Z"
        )
        
        assert source.dataset == "osm"
        assert source.property == "building"
        assert source.record_id == "123456"
        assert source.confidence == 0.95
        assert source.update_time == "2023-01-01T00:00:00Z"
    
    def test_empty_dataset_validation(self):
        """Test validation fails for empty dataset"""
        with pytest.raises(ValueError, match="Dataset cannot be empty"):
            SourceInfo(dataset="")
    
    def test_confidence_range_validation(self):
        """Test confidence value range validation"""
        # Valid confidence values
        SourceInfo(dataset="osm", confidence=0.0)
        SourceInfo(dataset="osm", confidence=0.5)
        SourceInfo(dataset="osm", confidence=1.0)
        
        # Invalid confidence values
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            SourceInfo(dataset="osm", confidence=-0.1)
        
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            SourceInfo(dataset="osm", confidence=1.1)
    
    def test_to_dict(self):
        """Test conversion to dictionary"""
        source = SourceInfo(
            dataset="osm",
            property="building",
            confidence=0.95
        )
        
        result = source.to_dict()
        expected = {
            "dataset": "osm",
            "property": "building",
            "record_id": None,
            "confidence": 0.95,
            "update_time": None
        }
        
        assert result == expected
    
    def test_from_dict(self):
        """Test creation from dictionary"""
        data = {
            "dataset": "osm",
            "property": "building",
            "confidence": 0.95
        }
        
        source = SourceInfo.from_dict(data)
        
        assert source.dataset == "osm"
        assert source.property == "building"
        assert source.confidence == 0.95


class TestFilterConfig:
    """Tests for FilterConfig class"""
    
    def test_empty_filter(self):
        """Test creating empty filter configuration"""
        filter_config = FilterConfig()
        
        assert filter_config.data_sources is None
        assert filter_config.min_confidence is None
        assert filter_config.max_confidence is None
        assert filter_config.metadata_filters is None
        assert filter_config.date_range is None
    
    def test_complete_filter(self):
        """Test creating complete filter configuration"""
        filter_config = FilterConfig(
            data_sources=["osm", "meta"],
            min_confidence=0.5,
            max_confidence=0.9,
            metadata_filters={"building_type": "residential"},
            date_range=("2023-01-01", "2023-12-31")
        )
        
        assert filter_config.data_sources == ["osm", "meta"]
        assert filter_config.min_confidence == 0.5
        assert filter_config.max_confidence == 0.9
        assert filter_config.metadata_filters == {"building_type": "residential"}
        assert filter_config.date_range == ("2023-01-01", "2023-12-31")
    
    def test_confidence_validation(self):
        """Test confidence range validation"""
        # Valid ranges
        FilterConfig(min_confidence=0.0, max_confidence=1.0)
        FilterConfig(min_confidence=0.5)
        FilterConfig(max_confidence=0.8)
        
        # Invalid ranges
        with pytest.raises(ValueError, match="min_confidence must be between 0.0 and 1.0"):
            FilterConfig(min_confidence=-0.1)
        
        with pytest.raises(ValueError, match="max_confidence must be between 0.0 and 1.0"):
            FilterConfig(max_confidence=1.1)
        
        with pytest.raises(ValueError, match="min_confidence cannot be greater than max_confidence"):
            FilterConfig(min_confidence=0.8, max_confidence=0.5)
    
    def test_matches_source(self):
        """Test source matching functionality"""
        filter_config = FilterConfig(
            data_sources=["osm"],
            min_confidence=0.5,
            max_confidence=0.9
        )
        
        # Matching source
        matching_source = SourceInfo(dataset="osm", confidence=0.7)
        assert filter_config.matches_source(matching_source)
        
        # Non-matching dataset
        wrong_dataset = SourceInfo(dataset="meta", confidence=0.7)
        assert not filter_config.matches_source(wrong_dataset)
        
        # Non-matching confidence (too low)
        low_confidence = SourceInfo(dataset="osm", confidence=0.3)
        assert not filter_config.matches_source(low_confidence)
        
        # Non-matching confidence (too high)
        high_confidence = SourceInfo(dataset="osm", confidence=0.95)
        assert not filter_config.matches_source(high_confidence)
        
        # Source without confidence (should match)
        no_confidence = SourceInfo(dataset="osm")
        assert filter_config.matches_source(no_confidence)


class TestAnalysisResult:
    """Tests for AnalysisResult class"""
    
    def test_valid_result(self):
        """Test creating a valid analysis result"""
        result = AnalysisResult(
            region_id="test_region",
            bounds=(-1.0, -1.0, 1.0, 1.0),
            feature_type="building",
            total_features=100,
            source_breakdown={"osm": 60, "meta": 40},
            coverage_metrics={"completeness": 0.95},
            quality_scores={"accuracy": 0.85}
        )
        
        assert result.region_id == "test_region"
        assert result.bounds == (-1.0, -1.0, 1.0, 1.0)
        assert result.feature_type == "building"
        assert result.total_features == 100
        assert result.source_breakdown == {"osm": 60, "meta": 40}
        assert isinstance(result.analysis_timestamp, datetime)
    
    def test_validation_errors(self):
        """Test various validation errors"""
        # Empty region ID
        with pytest.raises(ValueError, match="Region ID cannot be empty"):
            AnalysisResult(
                region_id="",
                bounds=(-1.0, -1.0, 1.0, 1.0),
                feature_type="building",
                total_features=100,
                source_breakdown={"osm": 100},
                coverage_metrics={},
                quality_scores={}
            )
        
        # Negative total features
        with pytest.raises(ValueError, match="Total features cannot be negative"):
            AnalysisResult(
                region_id="test",
                bounds=(-1.0, -1.0, 1.0, 1.0),
                feature_type="building",
                total_features=-1,
                source_breakdown={},
                coverage_metrics={},
                quality_scores={}
            )
        
        # Invalid bounds length
        with pytest.raises(ValueError, match="Bounds must be a tuple of 4 values"):
            AnalysisResult(
                region_id="test",
                bounds=(-1.0, -1.0, 1.0),  # Only 3 values
                feature_type="building",
                total_features=100,
                source_breakdown={"osm": 100},
                coverage_metrics={},
                quality_scores={}
            )
        
        # Invalid bounds values
        with pytest.raises(ValueError, match="Invalid bounds"):
            AnalysisResult(
                region_id="test",
                bounds=(1.0, 1.0, -1.0, -1.0),  # min > max
                feature_type="building",
                total_features=100,
                source_breakdown={"osm": 100},
                coverage_metrics={},
                quality_scores={}
            )
    
    def test_source_percentage(self):
        """Test source percentage calculation"""
        result = AnalysisResult(
            region_id="test",
            bounds=(-1.0, -1.0, 1.0, 1.0),
            feature_type="building",
            total_features=100,
            source_breakdown={"osm": 60, "meta": 40},
            coverage_metrics={},
            quality_scores={}
        )
        
        assert result.get_source_percentage("osm") == 60.0
        assert result.get_source_percentage("meta") == 40.0
        assert result.get_source_percentage("unknown") == 0.0
    
    def test_top_sources(self):
        """Test top sources functionality"""
        result = AnalysisResult(
            region_id="test",
            bounds=(-1.0, -1.0, 1.0, 1.0),
            feature_type="building",
            total_features=100,
            source_breakdown={"osm": 60, "meta": 30, "other": 10},
            coverage_metrics={},
            quality_scores={}
        )
        
        top_sources = result.get_top_sources(2)
        
        assert len(top_sources) == 2
        assert top_sources[0] == ("osm", 60, 60.0)
        assert top_sources[1] == ("meta", 30, 30.0)
    
    def test_to_from_dict(self):
        """Test dictionary conversion"""
        original = AnalysisResult(
            region_id="test",
            bounds=(-1.0, -1.0, 1.0, 1.0),
            feature_type="building",
            total_features=100,
            source_breakdown={"osm": 100},
            coverage_metrics={"completeness": 0.95},
            quality_scores={"accuracy": 0.85}
        )
        
        # Convert to dict and back
        data = original.to_dict()
        restored = AnalysisResult.from_dict(data)
        
        assert restored.region_id == original.region_id
        assert restored.bounds == original.bounds
        assert restored.feature_type == original.feature_type
        assert restored.total_features == original.total_features
        assert restored.source_breakdown == original.source_breakdown


class TestVisualizationConfig:
    """Tests for VisualizationConfig class"""
    
    def test_minimal_config(self):
        """Test creating minimal visualization configuration"""
        config = VisualizationConfig(chart_type="bar")
        
        assert config.chart_type == ChartType.BAR
        assert config.color_scheme == ColorScheme.DEFAULT
        assert config.interactive is True
        assert config.export_format is None
    
    def test_complete_config(self):
        """Test creating complete visualization configuration"""
        config = VisualizationConfig(
            chart_type="pie",
            color_scheme="viridis",
            interactive=False,
            export_format="png",
            title="Test Chart",
            width=800,
            height=600,
            show_legend=False,
            custom_colors={"osm": "#ff0000", "meta": "#00ff00"}
        )
        
        assert config.chart_type == ChartType.PIE
        assert config.color_scheme == ColorScheme.VIRIDIS
        assert config.interactive is False
        assert config.export_format == ExportFormat.PNG
        assert config.title == "Test Chart"
        assert config.width == 800
        assert config.height == 600
        assert config.show_legend is False
        assert config.custom_colors == {"osm": "#ff0000", "meta": "#00ff00"}
    
    def test_enum_validation(self):
        """Test enum validation"""
        # Valid enum values
        VisualizationConfig(chart_type="bar")
        VisualizationConfig(chart_type=ChartType.PIE)
        
        # Invalid enum values
        with pytest.raises(ValueError, match="Invalid chart_type"):
            VisualizationConfig(chart_type="invalid_type")
        
        with pytest.raises(ValueError, match="Invalid color_scheme"):
            VisualizationConfig(chart_type="bar", color_scheme="invalid_scheme")
        
        with pytest.raises(ValueError, match="Invalid export_format"):
            VisualizationConfig(chart_type="bar", export_format="invalid_format")
    
    def test_dimension_validation(self):
        """Test width and height validation"""
        # Valid dimensions
        VisualizationConfig(chart_type="bar", width=800, height=600)
        
        # Invalid dimensions
        with pytest.raises(ValueError, match="Width must be positive"):
            VisualizationConfig(chart_type="bar", width=0)
        
        with pytest.raises(ValueError, match="Height must be positive"):
            VisualizationConfig(chart_type="bar", height=-100)
    
    def test_to_from_dict(self):
        """Test dictionary conversion"""
        original = VisualizationConfig(
            chart_type="bar",
            color_scheme="viridis",
            title="Test Chart"
        )
        
        # Convert to dict and back
        data = original.to_dict()
        restored = VisualizationConfig.from_dict(data)
        
        assert restored.chart_type == original.chart_type
        assert restored.color_scheme == original.color_scheme
        assert restored.title == original.title


class TestAnalysisConfig:
    """Tests for AnalysisConfig class"""
    
    def test_valid_config(self):
        """Test creating valid analysis configuration"""
        feature_config = FeatureTypeConfig(
            theme="buildings",
            feature_type="building",
            base_path=Path("/tmp/test")
        )
        
        config = AnalysisConfig(
            feature_configs=[feature_config],
            bounds=(-1.0, -1.0, 1.0, 1.0),
            region_id="test_region"
        )
        
        assert len(config.feature_configs) == 1
        assert config.bounds == (-1.0, -1.0, 1.0, 1.0)
        assert config.region_id == "test_region"
    
    def test_validation_errors(self):
        """Test validation errors"""
        # Empty feature configs
        with pytest.raises(ValueError, match="At least one feature configuration is required"):
            AnalysisConfig(
                feature_configs=[],
                bounds=(-1.0, -1.0, 1.0, 1.0),
                region_id="test"
            )
        
        # Invalid max_features
        feature_config = FeatureTypeConfig(
            theme="buildings",
            feature_type="building",
            base_path=Path("/tmp/test")
        )
        
        with pytest.raises(ValueError, match="max_features must be positive"):
            AnalysisConfig(
                feature_configs=[feature_config],
                bounds=(-1.0, -1.0, 1.0, 1.0),
                region_id="test",
                max_features=0
            )


class TestUtilityFunctions:
    """Tests for utility functions"""
    
    def test_save_load_config(self):
        """Test saving and loading configuration files"""
        config = VisualizationConfig(
            chart_type="bar",
            title="Test Chart"
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            # Save and load
            save_config_to_file(config, temp_path)
            loaded_config = load_config_from_file(temp_path, "visualization")
            
            assert loaded_config.chart_type == config.chart_type
            assert loaded_config.title == config.title
        finally:
            temp_path.unlink()
    
    def test_schema_validation(self):
        """Test schema validation function"""
        # Valid data
        valid_feature_config = {
            "theme": "buildings",
            "feature_type": "building",
            "base_path": "/tmp/test"
        }
        
        assert validate_schema(valid_feature_config, "feature_config")
        
        # Invalid data
        invalid_feature_config = {
            "theme": "",  # Empty theme
            "feature_type": "building",
            "base_path": "/tmp/test"
        }
        
        assert not validate_schema(invalid_feature_config, "feature_config")
        
        # Unknown schema type
        with pytest.raises(ValueError, match="Unknown schema type"):
            validate_schema({}, "unknown_type")


if __name__ == "__main__":
    pytest.main([__file__])