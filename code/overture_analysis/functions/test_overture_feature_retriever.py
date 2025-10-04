"""Tests for the universal Overture feature retriever infrastructure."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from overture_feature_retriever import (
    OvertureFeatureRetriever,
    MultiFeatureRetriever,
    FeatureTypeInfo
)


class TestOvertureFeatureRetriever:
    """Test cases for OvertureFeatureRetriever class."""
    
    def test_get_available_themes(self, tmp_path):
        """Test theme discovery from directory structure."""
        # Create mock directory structure
        (tmp_path / "theme=buildings").mkdir()
        (tmp_path / "theme=places").mkdir()
        (tmp_path / "theme=transportation").mkdir()
        (tmp_path / "not_a_theme").mkdir()  # Should be ignored
        
        themes = OvertureFeatureRetriever.get_available_themes(tmp_path)
        
        assert sorted(themes) == ["buildings", "places", "transportation"]
    
    def test_get_available_types(self, tmp_path):
        """Test feature type discovery within a theme."""
        theme_path = tmp_path / "theme=buildings"
        theme_path.mkdir()
        (theme_path / "type=building").mkdir()
        (theme_path / "type=building_part").mkdir()
        (theme_path / "not_a_type").mkdir()  # Should be ignored
        
        types = OvertureFeatureRetriever.get_available_types(tmp_path, "buildings")
        
        assert sorted(types) == ["building", "building_part"]
    
    def test_get_available_types_nonexistent_theme(self, tmp_path):
        """Test behavior when theme doesn't exist."""
        types = OvertureFeatureRetriever.get_available_types(tmp_path, "nonexistent")
        assert types == []
    
    @patch('overture_feature_retriever.pq.ParquetFile')
    @patch('overture_feature_retriever.duckdb.connect')
    def test_initialization_success(self, mock_connect, mock_parquet_file, tmp_path):
        """Test successful initialization with valid data."""
        # Setup directory structure
        feature_path = tmp_path / "theme=buildings" / "type=building"
        feature_path.mkdir(parents=True)
        parquet_file = feature_path / "data.parquet"
        parquet_file.touch()
        
        # Mock parquet metadata
        mock_metadata = Mock()
        mock_metadata.metadata = {
            b'geo': b'{"primary_column": "geometry", "columns": {"geometry": {"bbox": [-180, -90, 180, 90]}}}'
        }
        mock_schema = Mock()
        mock_schema.names = ["id", "geometry", "sources"]
        mock_pf = Mock()
        mock_pf.metadata = mock_metadata
        mock_pf.schema_arrow = [
            Mock(name="id", type="string"),
            Mock(name="geometry", type="binary"),
            Mock(name="sources", type="list")
        ]
        mock_parquet_file.return_value = mock_pf
        
        # Mock DuckDB connection
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        mock_conn.execute.return_value = None
        
        # Test initialization
        retriever = OvertureFeatureRetriever(tmp_path, "buildings", "building")
        
        assert retriever.theme == "buildings"
        assert retriever.feature_type == "building"
        assert retriever.geometry_column == "geometry"
        assert retriever.file_count == 1
        assert "geometry_column" in retriever.schema_info
    
    def test_initialization_missing_path(self, tmp_path):
        """Test initialization with missing feature type path."""
        with pytest.raises(FileNotFoundError, match="Feature type path does not exist"):
            OvertureFeatureRetriever(tmp_path, "buildings", "building")


class TestMultiFeatureRetriever:
    """Test cases for MultiFeatureRetriever class."""
    
    def test_initialization(self, tmp_path):
        """Test MultiFeatureRetriever initialization."""
        multi_retriever = MultiFeatureRetriever(tmp_path)
        
        assert multi_retriever.base_path == tmp_path
        assert multi_retriever.get_feature_types() == []
    
    @patch('overture_feature_retriever.OvertureFeatureRetriever')
    def test_add_feature_type(self, mock_retriever_class, tmp_path):
        """Test adding feature types to multi-retriever."""
        mock_retriever = Mock()
        mock_retriever_class.return_value = mock_retriever
        
        multi_retriever = MultiFeatureRetriever(tmp_path)
        multi_retriever.add_feature_type("buildings", "building")
        
        assert ("buildings", "building") in multi_retriever.get_feature_types()
        mock_retriever_class.assert_called_once_with(
            tmp_path, "buildings", "building",
            auto_install_spatial=True,
            load_spatial=True
        )
    
    @patch('overture_feature_retriever.OvertureFeatureRetriever')
    def test_add_feature_type_failure(self, mock_retriever_class, tmp_path, capsys):
        """Test handling of failed feature type addition."""
        mock_retriever_class.side_effect = FileNotFoundError("Path not found")
        
        multi_retriever = MultiFeatureRetriever(tmp_path)
        multi_retriever.add_feature_type("buildings", "building")
        
        # Should not be added to retrievers
        assert multi_retriever.get_feature_types() == []
        
        # Should print warning
        captured = capsys.readouterr()
        assert "Warning: Could not add buildings/building" in captured.out
    
    @patch('overture_feature_retriever.OvertureFeatureRetriever')
    def test_remove_feature_type(self, mock_retriever_class, tmp_path):
        """Test removing feature types from multi-retriever."""
        mock_retriever = Mock()
        mock_retriever_class.return_value = mock_retriever
        
        multi_retriever = MultiFeatureRetriever(tmp_path)
        multi_retriever.add_feature_type("buildings", "building")
        
        assert ("buildings", "building") in multi_retriever.get_feature_types()
        
        multi_retriever.remove_feature_type("buildings", "building")
        
        assert multi_retriever.get_feature_types() == []
        mock_retriever.close.assert_called_once()
    
    @patch('overture_feature_retriever.OvertureFeatureRetriever')
    def test_query_all_types(self, mock_retriever_class, tmp_path):
        """Test querying all registered feature types."""
        # Setup mock retrievers
        mock_retriever1 = Mock()
        mock_retriever2 = Mock()
        
        df1 = pd.DataFrame({"id": [1, 2], "name": ["A", "B"]})
        df2 = pd.DataFrame({"id": [3, 4], "type": ["X", "Y"]})
        
        mock_retriever1.query_bbox.return_value = df1
        mock_retriever2.query_bbox.return_value = df2
        
        mock_retriever_class.side_effect = [mock_retriever1, mock_retriever2]
        
        # Setup multi-retriever
        multi_retriever = MultiFeatureRetriever(tmp_path)
        multi_retriever.add_feature_type("buildings", "building")
        multi_retriever.add_feature_type("places", "place")
        
        # Test query
        results = multi_retriever.query_all_types(-1, -1, 1, 1, limit=100)
        
        assert len(results) == 2
        assert ("buildings", "building") in results
        assert ("places", "place") in results
        
        pd.testing.assert_frame_equal(results[("buildings", "building")], df1)
        pd.testing.assert_frame_equal(results[("places", "place")], df2)
        
        # Verify query parameters were passed correctly
        mock_retriever1.query_bbox.assert_called_once_with(
            -1, -1, 1, 1, columns=None, limit=100
        )
        mock_retriever2.query_bbox.assert_called_once_with(
            -1, -1, 1, 1, columns=None, limit=100
        )
    
    @patch('overture_feature_retriever.OvertureFeatureRetriever')
    def test_get_schema_info(self, mock_retriever_class, tmp_path):
        """Test getting schema info for specific feature type."""
        mock_retriever = Mock()
        mock_retriever.schema_info = {
            "geometry_column": "geometry",
            "columns": {"id": "string", "geometry": "binary"},
            "theme": "buildings",
            "feature_type": "building"
        }
        mock_retriever_class.return_value = mock_retriever
        
        multi_retriever = MultiFeatureRetriever(tmp_path)
        multi_retriever.add_feature_type("buildings", "building")
        
        schema_info = multi_retriever.get_schema_info("buildings", "building")
        
        assert schema_info["theme"] == "buildings"
        assert schema_info["feature_type"] == "building"
        assert schema_info["geometry_column"] == "geometry"
    
    def test_get_schema_info_unregistered(self, tmp_path):
        """Test getting schema info for unregistered feature type."""
        multi_retriever = MultiFeatureRetriever(tmp_path)
        
        with pytest.raises(ValueError, match="Feature type buildings/building not registered"):
            multi_retriever.get_schema_info("buildings", "building")
    
    @patch('overture_feature_retriever.OvertureFeatureRetriever')
    def test_close(self, mock_retriever_class, tmp_path):
        """Test closing all retrievers."""
        mock_retriever1 = Mock()
        mock_retriever2 = Mock()
        mock_retriever_class.side_effect = [mock_retriever1, mock_retriever2]
        
        multi_retriever = MultiFeatureRetriever(tmp_path)
        multi_retriever.add_feature_type("buildings", "building")
        multi_retriever.add_feature_type("places", "place")
        
        multi_retriever.close()
        
        mock_retriever1.close.assert_called_once()
        mock_retriever2.close.assert_called_once()
        assert multi_retriever.get_feature_types() == []


if __name__ == "__main__":
    pytest.main([__file__])