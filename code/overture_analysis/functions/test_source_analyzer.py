"""Tests for the source analysis and extraction functionality."""

import pytest
import json
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from source_analyzer import (
    SourceAnalyzer,
    ComparativeSourceAnalyzer,
    SourceInfo,
    SourceStatistics
)


class TestSourceInfo:
    """Test cases for SourceInfo dataclass."""
    
    def test_source_info_creation(self):
        """Test creating SourceInfo objects."""
        source = SourceInfo(
            dataset="OpenStreetMap",
            property="building",
            record_id="way/123456",
            confidence=0.95,
            update_time="2023-01-01T00:00:00Z"
        )
        
        assert source.dataset == "OpenStreetMap"
        assert source.property == "building"
        assert source.record_id == "way/123456"
        assert source.confidence == 0.95
        assert source.update_time == "2023-01-01T00:00:00Z"
    
    def test_source_info_minimal(self):
        """Test creating SourceInfo with minimal data."""
        source = SourceInfo(dataset="TestDataset")
        
        assert source.dataset == "TestDataset"
        assert source.property is None
        assert source.record_id is None
        assert source.confidence is None
        assert source.update_time is None


class TestSourceAnalyzer:
    """Test cases for SourceAnalyzer class."""
    
    def test_initialization(self):
        """Test SourceAnalyzer initialization."""
        analyzer = SourceAnalyzer()
        assert analyzer.source_column == "sources"
        assert analyzer.confidence_threshold == 0.0
        
        analyzer_custom = SourceAnalyzer(
            source_column="custom_sources",
            confidence_threshold=0.5
        )
        assert analyzer_custom.source_column == "custom_sources"
        assert analyzer_custom.confidence_threshold == 0.5
    
    def test_extract_sources_empty_dataframe(self):
        """Test source extraction with empty DataFrame."""
        analyzer = SourceAnalyzer()
        df = pd.DataFrame()
        
        result = analyzer.extract_sources(df)
        
        assert result.empty
        assert list(result.columns) == [
            'feature_id', 'dataset', 'property', 'record_id', 'confidence', 'update_time'
        ]
    
    def test_extract_sources_missing_column(self):
        """Test source extraction when source column is missing."""
        analyzer = SourceAnalyzer()
        df = pd.DataFrame({'id': [1, 2], 'name': ['A', 'B']})
        
        result = analyzer.extract_sources(df)
        
        assert result.empty
        assert list(result.columns) == [
            'feature_id', 'dataset', 'property', 'record_id', 'confidence', 'update_time'
        ]
    
    def test_extract_sources_simple_strings(self):
        """Test source extraction with simple string datasets."""
        analyzer = SourceAnalyzer()
        df = pd.DataFrame({
            'id': ['feat1', 'feat2', 'feat3'],
            'sources': ['OpenStreetMap', 'Microsoft', None]
        })
        
        result = analyzer.extract_sources(df)
        
        assert len(result) == 2  # One None value should be skipped
        assert result.iloc[0]['feature_id'] == 'feat1'
        assert result.iloc[0]['dataset'] == 'OpenStreetMap'
        assert result.iloc[1]['feature_id'] == 'feat2'
        assert result.iloc[1]['dataset'] == 'Microsoft'
    
    def test_extract_sources_json_list(self):
        """Test source extraction with JSON list format."""
        analyzer = SourceAnalyzer()
        sources_json = json.dumps([
            {"dataset": "OpenStreetMap", "confidence": 0.9},
            {"dataset": "Microsoft", "property": "building", "confidence": 0.8}
        ])
        
        df = pd.DataFrame({
            'id': ['feat1'],
            'sources': [sources_json]
        })
        
        result = analyzer.extract_sources(df)
        
        assert len(result) == 2
        assert result.iloc[0]['dataset'] == 'OpenStreetMap'
        assert result.iloc[0]['confidence'] == 0.9
        assert result.iloc[1]['dataset'] == 'Microsoft'
        assert result.iloc[1]['property'] == 'building'
        assert result.iloc[1]['confidence'] == 0.8
    
    def test_extract_sources_dict_format(self):
        """Test source extraction with dictionary format."""
        analyzer = SourceAnalyzer()
        df = pd.DataFrame({
            'id': ['feat1'],
            'sources': [{"dataset": "TestSource", "record_id": "123", "confidence": 0.95}]
        })
        
        result = analyzer.extract_sources(df)
        
        assert len(result) == 1
        assert result.iloc[0]['dataset'] == 'TestSource'
        assert result.iloc[0]['record_id'] == '123'
        assert result.iloc[0]['confidence'] == 0.95
    
    def test_extract_sources_confidence_threshold(self):
        """Test source extraction with confidence threshold filtering."""
        analyzer = SourceAnalyzer(confidence_threshold=0.8)
        sources_json = json.dumps([
            {"dataset": "HighConfidence", "confidence": 0.9},
            {"dataset": "LowConfidence", "confidence": 0.5},
            {"dataset": "NoConfidence"}  # Should be included (no confidence score)
        ])
        
        df = pd.DataFrame({
            'id': ['feat1'],
            'sources': [sources_json]
        })
        
        result = analyzer.extract_sources(df)
        
        assert len(result) == 2  # Only high confidence and no confidence
        datasets = result['dataset'].tolist()
        assert 'HighConfidence' in datasets
        assert 'NoConfidence' in datasets
        assert 'LowConfidence' not in datasets
    
    def test_parse_sources_field_various_formats(self):
        """Test parsing of various source field formats."""
        analyzer = SourceAnalyzer()
        
        # Test string format
        result = analyzer._parse_sources_field("OpenStreetMap")
        assert len(result) == 1
        assert result[0].dataset == "OpenStreetMap"
        
        # Test JSON string format
        json_str = '{"dataset": "TestSource", "confidence": 0.8}'
        result = analyzer._parse_sources_field(json_str)
        assert len(result) == 1
        assert result[0].dataset == "TestSource"
        assert result[0].confidence == 0.8
        
        # Test list format
        source_list = [{"dataset": "Source1"}, {"dataset": "Source2"}]
        result = analyzer._parse_sources_field(source_list)
        assert len(result) == 2
        assert result[0].dataset == "Source1"
        assert result[1].dataset == "Source2"
        
        # Test None/NaN
        result = analyzer._parse_sources_field(None)
        assert len(result) == 0
        
        result = analyzer._parse_sources_field(pd.NA)
        assert len(result) == 0
    
    def test_parse_single_source_dict(self):
        """Test parsing single source dictionary with various field names."""
        analyzer = SourceAnalyzer()
        
        # Test standard field names
        source_dict = {
            "dataset": "OpenStreetMap",
            "property": "building",
            "record_id": "way/123",
            "confidence": 0.95,
            "update_time": "2023-01-01"
        }
        result = analyzer._parse_single_source(source_dict)
        assert len(result) == 1
        assert result[0].dataset == "OpenStreetMap"
        assert result[0].property == "building"
        assert result[0].record_id == "way/123"
        assert result[0].confidence == 0.95
        assert result[0].update_time == "2023-01-01"
        
        # Test alternative field names
        source_dict_alt = {
            "source": "Microsoft",
            "recordId": "12345",
            "updateTime": "2023-02-01"
        }
        result = analyzer._parse_single_source(source_dict_alt)
        assert len(result) == 1
        assert result[0].dataset == "Microsoft"
        assert result[0].record_id == "12345"
        assert result[0].update_time == "2023-02-01"
    
    def test_categorize_sources(self):
        """Test source categorization functionality."""
        analyzer = SourceAnalyzer()
        
        # Create test sources DataFrame
        sources_df = pd.DataFrame({
            'feature_id': ['f1', 'f2', 'f3', 'f4', 'f5'],
            'dataset': ['OpenStreetMap', 'Microsoft', 'OpenStreetMap', 'Google', 'Microsoft'],
            'property': [None, None, None, None, None],
            'record_id': [None, None, None, None, None],
            'confidence': [None, None, None, None, None],
            'update_time': [None, None, None, None, None]
        })
        
        result = analyzer.categorize_sources(sources_df)
        
        expected = {
            'OpenStreetMap': 2,
            'Microsoft': 2,
            'Google': 1
        }
        assert result == expected
    
    def test_categorize_sources_empty(self):
        """Test source categorization with empty DataFrame."""
        analyzer = SourceAnalyzer()
        sources_df = pd.DataFrame(columns=[
            'feature_id', 'dataset', 'property', 'record_id', 'confidence', 'update_time'
        ])
        
        result = analyzer.categorize_sources(sources_df)
        assert result == {}
    
    def test_calculate_coverage_metrics(self):
        """Test coverage metrics calculation."""
        analyzer = SourceAnalyzer()
        
        # Create test sources DataFrame
        sources_df = pd.DataFrame({
            'feature_id': ['f1', 'f2', 'f3', 'f4'],
            'dataset': ['OSM', 'Microsoft', 'OSM', 'Google'],
            'property': [None, None, None, None],
            'record_id': [None, None, None, None],
            'confidence': [0.9, None, 0.8, 0.95],
            'update_time': [None, None, None, None]
        })
        
        result = analyzer.calculate_coverage_metrics(sources_df)
        
        assert result['features_with_sources'] == 100.0
        assert result['avg_sources_per_feature'] == 1.0  # 4 sources / 4 features
        assert result['confidence_coverage'] == 75.0  # 3 out of 4 have confidence
        assert abs(result['avg_confidence'] - 0.883333) < 0.001  # (0.9 + 0.8 + 0.95) / 3
        assert result['dataset_diversity'] > 0  # Should have some diversity
    
    def test_calculate_coverage_metrics_empty(self):
        """Test coverage metrics with empty DataFrame."""
        analyzer = SourceAnalyzer()
        sources_df = pd.DataFrame(columns=[
            'feature_id', 'dataset', 'property', 'record_id', 'confidence', 'update_time'
        ])
        
        result = analyzer.calculate_coverage_metrics(sources_df)
        
        assert result['features_with_sources'] == 0.0
        assert result['avg_sources_per_feature'] == 0.0
        assert result['confidence_coverage'] == 0.0
        assert result['avg_confidence'] == 0.0
        assert result['dataset_diversity'] == 0.0
    
    def test_assess_data_quality(self):
        """Test data quality assessment."""
        analyzer = SourceAnalyzer()
        
        # Create test sources DataFrame with mixed completeness
        sources_df = pd.DataFrame({
            'feature_id': ['f1', 'f2', 'f3', 'f4'],
            'dataset': ['OSM', 'Microsoft', 'OSM', 'Google'],
            'property': ['building', None, 'amenity', None],
            'record_id': ['way/123', None, 'way/456', 'node/789'],
            'confidence': [0.9, None, 0.8, 0.95],
            'update_time': ['2023-01-01', None, None, '2023-02-01']
        })
        
        result = analyzer.assess_data_quality(sources_df)
        
        # Check completeness scores
        assert result['completeness_scores']['dataset'] == 100.0  # All have dataset
        assert result['completeness_scores']['property'] == 50.0  # 2 out of 4
        assert result['completeness_scores']['record_id'] == 75.0  # 3 out of 4
        assert result['completeness_scores']['confidence'] == 75.0  # 3 out of 4
        assert result['completeness_scores']['update_time'] == 50.0  # 2 out of 4
        
        # Check confidence distribution
        conf_dist = result['confidence_distribution']
        assert conf_dist['count'] == 3
        assert abs(conf_dist['mean'] - 0.883333) < 0.001
        assert conf_dist['min'] == 0.8
        assert conf_dist['max'] == 0.95
        
        # Check dataset consistency
        dataset_cons = result['dataset_consistency']
        assert dataset_cons['unique_datasets'] == 3
        assert dataset_cons['most_common_dataset'] == 'OSM'  # Appears twice
    
    def test_create_summary_statistics(self):
        """Test creation of comprehensive summary statistics."""
        analyzer = SourceAnalyzer()
        
        # Create test features DataFrame
        features_df = pd.DataFrame({
            'id': ['f1', 'f2', 'f3'],
            'sources': [
                '{"dataset": "OpenStreetMap", "confidence": 0.9}',
                '{"dataset": "Microsoft", "confidence": 0.8}',
                '{"dataset": "OpenStreetMap", "confidence": 0.95}'
            ]
        })
        
        result = analyzer.create_summary_statistics(features_df, "test_region")
        
        assert isinstance(result, SourceStatistics)
        assert result.total_features == 3
        assert result.total_sources == 3
        assert result.unique_datasets == 2
        assert result.source_breakdown['OpenStreetMap'] == 2
        assert result.source_breakdown['Microsoft'] == 1
        assert abs(result.source_percentages['OpenStreetMap'] - 66.666667) < 0.001
        assert abs(result.source_percentages['Microsoft'] - 33.333333) < 0.001


class TestComparativeSourceAnalyzer:
    """Test cases for ComparativeSourceAnalyzer class."""
    
    def test_initialization(self):
        """Test ComparativeSourceAnalyzer initialization."""
        analyzer = ComparativeSourceAnalyzer()
        assert analyzer.base_analyzer is not None
        assert isinstance(analyzer.base_analyzer, SourceAnalyzer)
        
        # Test with custom base analyzer
        custom_analyzer = SourceAnalyzer(confidence_threshold=0.5)
        comp_analyzer = ComparativeSourceAnalyzer(custom_analyzer)
        assert comp_analyzer.base_analyzer is custom_analyzer
    
    def test_compare_regions(self):
        """Test regional comparison functionality."""
        analyzer = ComparativeSourceAnalyzer()
        
        # Create test data for multiple regions
        region1_df = pd.DataFrame({
            'id': ['f1', 'f2'],
            'sources': ['OpenStreetMap', 'Microsoft']
        })
        
        region2_df = pd.DataFrame({
            'id': ['f3', 'f4', 'f5'],
            'sources': ['OpenStreetMap', 'OpenStreetMap', 'Google']
        })
        
        region_analyses = [
            {'region_id': 'region1', 'features_df': region1_df},
            {'region_id': 'region2', 'features_df': region2_df}
        ]
        
        result = analyzer.compare_regions(region_analyses)
        
        assert len(result) == 2
        assert 'region_id' in result.columns
        assert 'total_features' in result.columns
        assert 'OpenStreetMap' in result.columns
        assert 'Microsoft' in result.columns
        assert 'Google' in result.columns
        
        # Check region1 data
        region1_row = result[result['region_id'] == 'region1'].iloc[0]
        assert region1_row['total_features'] == 2
        assert region1_row['OpenStreetMap'] == 1
        assert region1_row['Microsoft'] == 1
        assert region1_row['Google'] == 0  # Should be filled with 0
        
        # Check region2 data
        region2_row = result[result['region_id'] == 'region2'].iloc[0]
        assert region2_row['total_features'] == 3
        assert region2_row['OpenStreetMap'] == 2
        assert region2_row['Microsoft'] == 0  # Should be filled with 0
        assert region2_row['Google'] == 1
    
    def test_compare_regions_empty(self):
        """Test regional comparison with empty input."""
        analyzer = ComparativeSourceAnalyzer()
        result = analyzer.compare_regions([])
        assert result.empty
    
    def test_compare_feature_types(self):
        """Test feature type comparison functionality."""
        analyzer = ComparativeSourceAnalyzer()
        
        # Create test data for different feature types
        buildings_df = pd.DataFrame({
            'id': ['b1', 'b2'],
            'sources': ['OpenStreetMap', 'Microsoft']
        })
        
        places_df = pd.DataFrame({
            'id': ['p1', 'p2', 'p3'],
            'sources': ['OpenStreetMap', 'Google', 'Foursquare']
        })
        
        type_analyses = {
            ('buildings', 'building'): {'features_df': buildings_df},
            ('places', 'place'): {'features_df': places_df}
        }
        
        result = analyzer.compare_feature_types(type_analyses)
        
        assert len(result) == 2
        assert 'theme' in result.columns
        assert 'feature_type' in result.columns
        assert 'type_id' in result.columns
        
        # Check buildings data
        buildings_row = result[result['theme'] == 'buildings'].iloc[0]
        assert buildings_row['feature_type'] == 'building'
        assert buildings_row['type_id'] == 'buildings/building'
        assert buildings_row['total_features'] == 2
        assert buildings_row['OpenStreetMap'] == 1
        assert buildings_row['Microsoft'] == 1
        
        # Check places data
        places_row = result[result['theme'] == 'places'].iloc[0]
        assert places_row['feature_type'] == 'place'
        assert places_row['type_id'] == 'places/place'
        assert places_row['total_features'] == 3
        assert places_row['OpenStreetMap'] == 1
        assert places_row['Google'] == 1
        assert places_row['Foursquare'] == 1
    
    def test_identify_coverage_gaps(self):
        """Test coverage gap identification."""
        analyzer = ComparativeSourceAnalyzer()
        
        # Create test data with coverage gaps
        region1_df = pd.DataFrame({
            'id': ['f1', 'f2', 'f3', 'f4', 'f5'],  # 5 features
            'sources': ['OpenStreetMap', 'OpenStreetMap', 'OpenStreetMap', 'OpenStreetMap', 'Microsoft']
            # OSM: 4/5 = 80%, Microsoft: 1/5 = 20%
        })
        
        region2_df = pd.DataFrame({
            'id': ['f6', 'f7', 'f8', 'f9', 'f10'],  # 5 features
            'sources': ['Google', 'Google', 'Google', 'Google', 'Google']
            # Google: 5/5 = 100%, OSM: 0/5 = 0%, Microsoft: 0/5 = 0%
        })
        
        analyses = [
            {'region_id': 'region1', 'features_df': region1_df},
            {'region_id': 'region2', 'features_df': region2_df}
        ]
        
        # Test with 50% threshold - should find gaps
        gaps = analyzer.identify_coverage_gaps(analyses, min_coverage_threshold=50.0)
        
        # Should find gaps for Microsoft in region1 (20%) and OSM/Microsoft in region2 (0%)
        assert len(gaps) >= 3
        
        # Check that gaps are sorted by coverage percentage
        coverage_percentages = [gap['coverage_percentage'] for gap in gaps]
        assert coverage_percentages == sorted(coverage_percentages)
        
        # Check specific gaps
        gap_descriptions = [(gap['region_id'], gap['dataset'], gap['gap_type']) for gap in gaps]
        assert ('region2', 'OpenStreetMap', 'missing_dataset') in gap_descriptions
        assert ('region2', 'Microsoft', 'missing_dataset') in gap_descriptions
    
    def test_identify_coverage_gaps_empty(self):
        """Test coverage gap identification with empty input."""
        analyzer = ComparativeSourceAnalyzer()
        gaps = analyzer.identify_coverage_gaps([])
        assert gaps == []


if __name__ == "__main__":
    pytest.main([__file__])