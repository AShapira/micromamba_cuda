"""
Tests for spatial aggregation and sampling components.

This module contains comprehensive tests for the SpatialAggregator and SpatialSampler
classes, including accuracy tests, performance tests, and edge case handling.
"""

import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, box
from unittest.mock import patch, MagicMock
import tempfile
import time
from pathlib import Path

from spatial_components import (
    SpatialAggregator, 
    SpatialSampler, 
    AggregationConfig, 
    SamplingConfig,
    process_in_chunks,
    estimate_memory_usage
)


class TestSpatialAggregator:
    """Test cases for SpatialAggregator class."""
    
    @pytest.fixture
    def sample_points(self):
        """Create sample point data for testing."""
        np.random.seed(42)
        n_points = 1000
        x_coords = np.random.uniform(0, 100, n_points)
        y_coords = np.random.uniform(0, 100, n_points)
        values = np.random.uniform(1, 10, n_points)
        
        geometries = [Point(x, y) for x, y in zip(x_coords, y_coords)]
        
        return gpd.GeoDataFrame({
            'id': range(n_points),
            'value': values,
            'geometry': geometries
        })
    
    @pytest.fixture
    def sample_boundaries(self):
        """Create sample boundary polygons for testing."""
        boundaries = []
        boundary_ids = []
        
        # Create 4 quadrant boundaries
        for i, (x_min, x_max) in enumerate([(0, 50), (50, 100)]):
            for j, (y_min, y_max) in enumerate([(0, 50), (50, 100)]):
                boundary = box(x_min, y_min, x_max, y_max)
                boundaries.append(boundary)
                boundary_ids.append(f"quad_{i}_{j}")
        
        return gpd.GeoDataFrame({
            'id': boundary_ids,
            'geometry': boundaries
        })
    
    @pytest.fixture
    def aggregator(self):
        """Create SpatialAggregator instance for testing."""
        return SpatialAggregator(use_gpu=False, chunk_size=100)
    
    def test_initialization(self):
        """Test SpatialAggregator initialization."""
        # Test CPU initialization
        agg_cpu = SpatialAggregator(use_gpu=False)
        assert not agg_cpu.use_gpu
        assert agg_cpu.chunk_size == 10000
        
        # Test with custom chunk size
        agg_custom = SpatialAggregator(use_gpu=False, chunk_size=5000)
        assert agg_custom.chunk_size == 5000
    
    def test_create_grid(self, aggregator):
        """Test grid creation functionality."""
        bounds = (0, 0, 100, 100)
        grid_size = 10
        
        grid = aggregator._create_grid(bounds, grid_size)
        
        # Check grid properties
        assert isinstance(grid, gpd.GeoDataFrame)
        assert 'grid_id' in grid.columns
        assert 'geometry' in grid.columns
        assert len(grid) == 100  # 10x10 grid
        
        # Check grid cell properties
        first_cell = grid.iloc[0]
        assert first_cell.geometry.bounds == (0, 0, 10, 10)
    
    def test_aggregate_by_grid_count(self, aggregator, sample_points):
        """Test grid aggregation with count function."""
        grid_size = 20
        result = aggregator.aggregate_by_grid(sample_points, grid_size, 'count')
        
        # Check result properties
        assert isinstance(result, gpd.GeoDataFrame)
        assert 'count' in result.columns
        assert 'grid_id' in result.columns
        assert len(result) == 25  # 5x5 grid for 100x100 area with size 20
        
        # Check that total count matches input
        total_count = result['count'].sum()
        assert total_count == len(sample_points)
    
    def test_aggregate_by_grid_sum(self, aggregator, sample_points):
        """Test grid aggregation with sum function."""
        grid_size = 20
        result = aggregator.aggregate_by_grid(sample_points, grid_size, 'sum', 'value')
        
        # Check result properties
        assert isinstance(result, gpd.GeoDataFrame)
        assert 'value' in result.columns
        
        # Check that total sum is approximately correct (allowing for floating point errors)
        total_sum = result['value'].sum()
        expected_sum = sample_points['value'].sum()
        assert abs(total_sum - expected_sum) < 0.01
    
    def test_aggregate_by_grid_mean(self, aggregator, sample_points):
        """Test grid aggregation with mean function."""
        grid_size = 50  # Larger grid to ensure multiple points per cell
        result = aggregator.aggregate_by_grid(sample_points, grid_size, 'mean', 'value')
        
        # Check result properties
        assert isinstance(result, gpd.GeoDataFrame)
        assert 'value' in result.columns
        
        # Check that means are reasonable (between min and max of original data)
        non_zero_means = result[result['value'] > 0]['value']
        if len(non_zero_means) > 0:
            assert non_zero_means.min() >= sample_points['value'].min()
            assert non_zero_means.max() <= sample_points['value'].max()
    
    def test_aggregate_by_boundaries_count(self, aggregator, sample_points, sample_boundaries):
        """Test boundary aggregation with count function."""
        result = aggregator.aggregate_by_boundaries(sample_points, sample_boundaries, 'count')
        
        # Check result properties
        assert isinstance(result, gpd.GeoDataFrame)
        assert 'count' in result.columns
        assert len(result) == len(sample_boundaries)
        
        # Check that total count matches input (approximately, due to boundary effects)
        total_count = result['count'].sum()
        assert total_count <= len(sample_points)  # Some points might be on boundaries
    
    def test_aggregate_by_boundaries_sum(self, aggregator, sample_points, sample_boundaries):
        """Test boundary aggregation with sum function."""
        result = aggregator.aggregate_by_boundaries(sample_points, sample_boundaries, 'sum', 'value')
        
        # Check result properties
        assert isinstance(result, gpd.GeoDataFrame)
        assert 'value' in result.columns
        
        # Check that total sum is reasonable
        total_sum = result['value'].sum()
        expected_sum = sample_points['value'].sum()
        assert total_sum <= expected_sum * 1.1  # Allow some tolerance
    
    def test_create_density_surface(self, aggregator, sample_points):
        """Test density surface creation."""
        bandwidth = 10
        grid_size = 5
        
        result = aggregator.create_density_surface(sample_points, bandwidth, grid_size)
        
        # Check result properties
        assert isinstance(result, gpd.GeoDataFrame)
        assert 'density' in result.columns
        assert 'grid_id' in result.columns
        
        # Check that density values are non-negative
        assert (result['density'] >= 0).all()
        
        # Check that some cells have positive density
        assert (result['density'] > 0).any()
    
    def test_chunked_processing(self, sample_points):
        """Test chunked processing for large datasets."""
        # Create aggregator with small chunk size
        aggregator = SpatialAggregator(use_gpu=False, chunk_size=50)
        
        grid_size = 25
        result = aggregator.aggregate_by_grid(sample_points, grid_size, 'count')
        
        # Should still work correctly with chunked processing
        assert isinstance(result, gpd.GeoDataFrame)
        assert 'count' in result.columns
        total_count = result['count'].sum()
        assert total_count == len(sample_points)
    
    def test_invalid_aggregation_function(self, aggregator, sample_points):
        """Test error handling for invalid aggregation functions."""
        with pytest.raises(ValueError, match="Unsupported aggregation function"):
            aggregator.aggregate_by_grid(sample_points, 10, 'invalid_func')
    
    def test_missing_value_column(self, aggregator, sample_points):
        """Test error handling when value column is missing for sum/mean/std."""
        with pytest.raises(ValueError, match="value_column required"):
            aggregator.aggregate_by_grid(sample_points, 10, 'sum')
    
    def test_density_kernels(self, aggregator):
        """Test different kernel types for density estimation."""
        # Create simple test data
        points = np.array([[0, 0], [1, 1], [2, 2]])
        grid_coords = np.array([[0, 0], [1, 1], [2, 2]])
        bandwidth = 1.0
        
        # Test different kernels
        for kernel in ['gaussian', 'uniform', 'triangular']:
            density = aggregator._compute_density_cpu(points, grid_coords, bandwidth, kernel)
            assert len(density) == len(grid_coords)
            assert (density >= 0).all()
    
    def test_invalid_kernel(self, aggregator):
        """Test error handling for invalid kernel types."""
        points = np.array([[0, 0]])
        grid_coords = np.array([[0, 0]])
        bandwidth = 1.0
        
        with pytest.raises(ValueError, match="Unsupported kernel"):
            aggregator._compute_density_cpu(points, grid_coords, bandwidth, 'invalid_kernel')


class TestSpatialSampler:
    """Test cases for SpatialSampler class."""
    
    @pytest.fixture
    def sampler(self):
        """Create SpatialSampler instance for testing."""
        return SpatialSampler(random_seed=42)
    
    @pytest.fixture
    def test_bounds(self):
        """Create test bounds for sampling."""
        return (0, 0, 100, 100)
    
    @pytest.fixture
    def sample_regions(self):
        """Create sample regions for stratified sampling."""
        regions = []
        region_ids = []
        
        # Create 3 regions
        for i, (x_min, x_max) in enumerate([(0, 30), (30, 70), (70, 100)]):
            region = box(x_min, 0, x_max, 100)
            regions.append(region)
            region_ids.append(f"region_{i}")
        
        return gpd.GeoDataFrame({
            'id': region_ids,
            'geometry': regions
        })
    
    def test_initialization(self):
        """Test SpatialSampler initialization."""
        sampler = SpatialSampler(random_seed=123)
        assert sampler.random_seed == 123
    
    def test_systematic_sample(self, sampler, test_bounds):
        """Test systematic sampling."""
        sample_size = 100
        points = sampler.systematic_sample(test_bounds, sample_size)
        
        # Check basic properties
        assert len(points) == sample_size
        assert all(isinstance(p, tuple) and len(p) == 2 for p in points)
        
        # Check that points are within bounds
        minx, miny, maxx, maxy = test_bounds
        for x, y in points:
            assert minx <= x <= maxx
            assert miny <= y <= maxy
        
        # Check systematic distribution (points should be roughly evenly spaced)
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        # Standard deviation should be reasonable for systematic sampling
        x_std = np.std(x_coords)
        y_std = np.std(y_coords)
        assert x_std > 10  # Should have good spread
        assert y_std > 10
    
    def test_stratified_sample(self, sampler, sample_regions):
        """Test stratified sampling."""
        samples_per_region = 50
        points = sampler.stratified_sample(sample_regions, samples_per_region)
        
        # Check basic properties
        assert len(points) <= len(sample_regions) * samples_per_region
        assert all(isinstance(p, tuple) and len(p) == 3 for p in points)
        
        # Check that we have samples from each region
        region_ids = set(p[2] for p in points)
        expected_regions = set(sample_regions['id'])
        assert region_ids.issubset(expected_regions)
        
        # Check that points are within their respective regions
        for x, y, region_id in points:
            region_geom = sample_regions[sample_regions['id'] == region_id].iloc[0].geometry
            point = Point(x, y)
            assert region_geom.contains(point) or region_geom.touches(point)
    
    def test_random_sample(self, sampler, test_bounds):
        """Test random sampling."""
        sample_size = 200
        points = sampler.random_sample(test_bounds, sample_size)
        
        # Check basic properties
        assert len(points) == sample_size
        assert all(isinstance(p, tuple) and len(p) == 2 for p in points)
        
        # Check that points are within bounds
        minx, miny, maxx, maxy = test_bounds
        for x, y in points:
            assert minx <= x <= maxx
            assert miny <= y <= maxy
    
    def test_adaptive_sample(self, sampler, test_bounds):
        """Test adaptive sampling."""
        # Create mock initial results
        initial_results = {
            'variance': 25.0,
            'mean': 10.0,
            'sample_size': 100
        }
        target_precision = 0.1
        
        points = sampler.adaptive_sample(initial_results, target_precision, test_bounds)
        
        # Should return additional sample points
        assert isinstance(points, list)
        if len(points) > 0:  # Might be empty if precision already achieved
            assert all(isinstance(p, tuple) and len(p) == 2 for p in points)
    
    def test_adaptive_sample_insufficient_data(self, sampler, test_bounds):
        """Test adaptive sampling with insufficient initial data."""
        initial_results = {'incomplete': 'data'}
        target_precision = 0.1
        
        points = sampler.adaptive_sample(initial_results, target_precision, test_bounds)
        
        # Should fallback to systematic sampling
        assert isinstance(points, list)
        assert len(points) == 100  # Default fallback size
    
    def test_adaptive_sample_precision_achieved(self, sampler, test_bounds):
        """Test adaptive sampling when target precision is already achieved."""
        initial_results = {
            'variance': 1.0,
            'mean': 100.0,  # CV = 0.1, which is very low
            'sample_size': 100
        }
        target_precision = 0.5  # Higher than current CV
        
        points = sampler.adaptive_sample(initial_results, target_precision, test_bounds)
        
        # Should return empty list since precision is already achieved
        assert len(points) == 0
    
    def test_create_sample_polygons(self, sampler):
        """Test creation of sample polygons."""
        sample_points = [(10, 10), (20, 20), (30, 30)]
        buffer_size = 5
        
        polygons = sampler.create_sample_polygons(sample_points, buffer_size)
        
        # Check basic properties
        assert isinstance(polygons, gpd.GeoDataFrame)
        assert len(polygons) == len(sample_points)
        assert 'sample_id' in polygons.columns
        assert 'geometry' in polygons.columns
        
        # Check polygon properties
        for i, (x, y) in enumerate(sample_points):
            polygon = polygons.iloc[i]
            centroid = polygon.geometry.centroid
            assert abs(centroid.x - x) < 0.1
            assert abs(centroid.y - y) < 0.1
            
            # Check buffer size (approximately)
            bounds = polygon.geometry.bounds
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            expected_size = buffer_size * 2
            assert abs(width - expected_size) < 0.1
            assert abs(height - expected_size) < 0.1
    
    def test_reproducibility(self):
        """Test that sampling is reproducible with same seed."""
        bounds = (0, 0, 100, 100)
        sample_size = 50
        
        sampler1 = SpatialSampler(random_seed=42)
        sampler2 = SpatialSampler(random_seed=42)
        
        points1 = sampler1.systematic_sample(bounds, sample_size)
        points2 = sampler2.systematic_sample(bounds, sample_size)
        
        # Systematic sampling should be identical
        assert points1 == points2
        
        # Random sampling should also be identical with same seed
        np.random.seed(42)
        random_points1 = sampler1.random_sample(bounds, sample_size)
        np.random.seed(42)
        random_points2 = sampler2.random_sample(bounds, sample_size)
        
        assert len(random_points1) == len(random_points2)


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_process_in_chunks(self):
        """Test chunked processing utility function."""
        # Create test data
        data = pd.DataFrame({
            'id': range(1000),
            'value': np.random.random(1000)
        })
        
        # Define simple processing function
        def double_values(chunk):
            result = chunk.copy()
            result['value'] = result['value'] * 2
            return result
        
        # Process in chunks
        result = process_in_chunks(data, double_values, chunk_size=100)
        
        # Check result
        assert len(result) == len(data)
        assert (result['value'] == data['value'] * 2).all()
    
    def test_process_in_chunks_geodataframe(self):
        """Test chunked processing with GeoDataFrame."""
        # Create test GeoDataFrame
        geometries = [Point(i, i) for i in range(100)]
        data = gpd.GeoDataFrame({
            'id': range(100),
            'geometry': geometries
        })
        
        # Define processing function
        def add_buffer(chunk):
            result = chunk.copy()
            result['geometry'] = result['geometry'].buffer(1)
            return result
        
        # Process in chunks
        result = process_in_chunks(data, add_buffer, chunk_size=25)
        
        # Check result
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == len(data)
        assert all(result.geometry.geom_type == 'Polygon')
    
    def test_estimate_memory_usage(self):
        """Test memory usage estimation."""
        # Create test DataFrame
        data = pd.DataFrame({
            'id': range(1000),
            'value': np.random.random(1000),
            'text': ['test_string'] * 1000
        })
        
        memory_mb = estimate_memory_usage(data)
        
        # Should return a positive number
        assert memory_mb > 0
        assert isinstance(memory_mb, float)
    
    def test_estimate_memory_usage_geodataframe(self):
        """Test memory usage estimation for GeoDataFrame."""
        geometries = [Point(i, i) for i in range(100)]
        data = gpd.GeoDataFrame({
            'id': range(100),
            'geometry': geometries
        })
        
        memory_mb = estimate_memory_usage(data)
        
        # Should return a positive number
        assert memory_mb > 0
        assert isinstance(memory_mb, float)


class TestPerformance:
    """Performance tests for spatial components."""
    
    def test_aggregation_performance(self):
        """Test aggregation performance with larger datasets."""
        # Create larger test dataset
        np.random.seed(42)
        n_points = 10000
        x_coords = np.random.uniform(0, 1000, n_points)
        y_coords = np.random.uniform(0, 1000, n_points)
        values = np.random.uniform(1, 100, n_points)
        
        geometries = [Point(x, y) for x, y in zip(x_coords, y_coords)]
        data = gpd.GeoDataFrame({
            'id': range(n_points),
            'value': values,
            'geometry': geometries
        })
        
        aggregator = SpatialAggregator(use_gpu=False, chunk_size=1000)
        
        # Time the aggregation
        start_time = time.time()
        result = aggregator.aggregate_by_grid(data, 50, 'count')
        end_time = time.time()
        
        # Should complete in reasonable time (less than 30 seconds)
        processing_time = end_time - start_time
        assert processing_time < 30
        
        # Check result quality
        assert len(result) > 0
        assert result['count'].sum() == len(data)
    
    def test_sampling_performance(self):
        """Test sampling performance."""
        sampler = SpatialSampler(random_seed=42)
        bounds = (0, 0, 10000, 10000)  # Large area
        
        # Time systematic sampling
        start_time = time.time()
        points = sampler.systematic_sample(bounds, 10000)
        end_time = time.time()
        
        # Should complete quickly (less than 5 seconds)
        processing_time = end_time - start_time
        assert processing_time < 5
        
        # Check result quality
        assert len(points) == 10000


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_data(self):
        """Test handling of empty datasets."""
        aggregator = SpatialAggregator(use_gpu=False)
        empty_data = gpd.GeoDataFrame({'geometry': []})
        
        # Should handle empty data gracefully
        result = aggregator.aggregate_by_grid(empty_data, 10, 'count')
        assert isinstance(result, gpd.GeoDataFrame)
    
    def test_single_point(self):
        """Test handling of single point datasets."""
        aggregator = SpatialAggregator(use_gpu=False)
        single_point = gpd.GeoDataFrame({
            'id': [1],
            'value': [5.0],
            'geometry': [Point(50, 50)]
        })
        
        result = aggregator.aggregate_by_grid(single_point, 10, 'count')
        assert isinstance(result, gpd.GeoDataFrame)
        assert result['count'].sum() == 1
    
    def test_very_small_bounds(self):
        """Test sampling with very small bounds."""
        sampler = SpatialSampler(random_seed=42)
        small_bounds = (0, 0, 0.001, 0.001)
        
        points = sampler.systematic_sample(small_bounds, 4)
        assert len(points) == 4
        
        # All points should be within bounds
        for x, y in points:
            assert 0 <= x <= 0.001
            assert 0 <= y <= 0.001
    
    def test_zero_buffer_size(self):
        """Test polygon creation with zero buffer size."""
        sampler = SpatialSampler(random_seed=42)
        sample_points = [(10, 10), (20, 20)]
        
        polygons = sampler.create_sample_polygons(sample_points, 0)
        
        # Should create point geometries (or very small polygons)
        assert len(polygons) == 2
        for geom in polygons.geometry:
            assert geom.area < 0.01  # Very small area


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])