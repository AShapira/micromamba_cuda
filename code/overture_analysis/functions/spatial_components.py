"""
Spatial aggregation and sampling components for the Overture Data Source Analyzer.

This module provides classes for spatial aggregation and sampling strategies
to handle large datasets efficiently while maintaining spatial accuracy.
"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
import warnings

# Try to import RAPIDS components for GPU acceleration
try:
    import cudf
    import cupy as cp
    RAPIDS_AVAILABLE = True
except ImportError:
    RAPIDS_AVAILABLE = False
    warnings.warn("RAPIDS not available, falling back to CPU processing")

logger = logging.getLogger(__name__)


@dataclass
class AggregationConfig:
    """Configuration for spatial aggregation operations"""
    method: str  # 'grid', 'boundaries', 'density'
    grid_size: Optional[float] = None
    boundaries: Optional[gpd.GeoDataFrame] = None
    density_bandwidth: Optional[float] = None
    chunk_size: int = 10000
    use_gpu: bool = True


@dataclass
class SamplingConfig:
    """Configuration for spatial sampling operations"""
    method: str  # 'systematic', 'stratified', 'adaptive'
    sample_size: int = 1000
    samples_per_region: Optional[int] = None
    target_precision: Optional[float] = None
    random_seed: int = 42


class SpatialAggregator:
    """
    Aggregates spatial data at different scales with support for large datasets
    and GPU acceleration when available.
    """
    
    def __init__(self, use_gpu: bool = True, chunk_size: int = 10000):
        """
        Initialize the SpatialAggregator.
        
        Args:
            use_gpu: Whether to use GPU acceleration if available
            chunk_size: Size of chunks for processing large datasets
        """
        self.use_gpu = use_gpu and RAPIDS_AVAILABLE
        self.chunk_size = chunk_size
        self._cache = {}
        
        if self.use_gpu:
            logger.info("SpatialAggregator initialized with GPU acceleration")
        else:
            logger.info("SpatialAggregator initialized with CPU processing")
    
    def aggregate_by_grid(
        self, 
        data: gpd.GeoDataFrame, 
        grid_size: float,
        aggregation_func: str = 'count',
        value_column: Optional[str] = None
    ) -> gpd.GeoDataFrame:
        """
        Aggregate spatial data into a regular grid.
        
        Args:
            data: Input GeoDataFrame
            grid_size: Size of grid cells in the data's CRS units
            aggregation_func: Aggregation function ('count', 'sum', 'mean', 'std')
            value_column: Column to aggregate (required for sum, mean, std)
            
        Returns:
            GeoDataFrame with grid cells and aggregated values
        """
        logger.info(f"Aggregating {len(data)} features by grid (size: {grid_size})")
        
        # Get bounds and create grid
        bounds = data.total_bounds
        grid = self._create_grid(bounds, grid_size)
        
        # Process in chunks if dataset is large
        if len(data) > self.chunk_size:
            return self._aggregate_by_grid_chunked(data, grid, aggregation_func, value_column)
        
        # Perform spatial join
        joined = gpd.sjoin(grid, data, how='left', predicate='intersects')
        
        # Aggregate based on function
        if aggregation_func == 'count':
            result = joined.groupby('grid_id').size().reset_index(name='count')
        elif aggregation_func in ['sum', 'mean', 'std']:
            if value_column is None:
                raise ValueError(f"value_column required for aggregation function '{aggregation_func}'")
            if aggregation_func == 'sum':
                result = joined.groupby('grid_id')[value_column].sum().reset_index()
            elif aggregation_func == 'mean':
                result = joined.groupby('grid_id')[value_column].mean().reset_index()
            elif aggregation_func == 'std':
                result = joined.groupby('grid_id')[value_column].std().reset_index()
        else:
            raise ValueError(f"Unsupported aggregation function: {aggregation_func}")
        
        # Merge back with grid geometry
        grid_result = grid.merge(result, on='grid_id', how='left')
        grid_result = grid_result.fillna(0)
        
        logger.info(f"Created grid with {len(grid_result)} cells")
        return grid_result
    
    def aggregate_by_boundaries(
        self, 
        data: gpd.GeoDataFrame, 
        boundaries: gpd.GeoDataFrame,
        aggregation_func: str = 'count',
        value_column: Optional[str] = None,
        boundary_id_column: str = 'id'
    ) -> gpd.GeoDataFrame:
        """
        Aggregate spatial data by administrative or custom boundaries.
        
        Args:
            data: Input GeoDataFrame
            boundaries: GeoDataFrame with boundary polygons
            aggregation_func: Aggregation function ('count', 'sum', 'mean', 'std')
            value_column: Column to aggregate (required for sum, mean, std)
            boundary_id_column: Column name for boundary identifiers
            
        Returns:
            GeoDataFrame with boundaries and aggregated values
        """
        logger.info(f"Aggregating {len(data)} features by {len(boundaries)} boundaries")
        
        # Ensure boundaries have an ID column
        if boundary_id_column not in boundaries.columns:
            boundaries = boundaries.copy()
            boundaries[boundary_id_column] = range(len(boundaries))
        
        # Process in chunks if dataset is large
        if len(data) > self.chunk_size:
            return self._aggregate_by_boundaries_chunked(
                data, boundaries, aggregation_func, value_column, boundary_id_column
            )
        
        # Perform spatial join
        joined = gpd.sjoin(data, boundaries, how='inner', predicate='intersects')
        
        # Aggregate based on function
        if aggregation_func == 'count':
            result = joined.groupby(f'{boundary_id_column}_right').size().reset_index(name='count')
            result = result.rename(columns={f'{boundary_id_column}_right': boundary_id_column})
        elif aggregation_func in ['sum', 'mean', 'std']:
            if value_column is None:
                raise ValueError(f"value_column required for aggregation function '{aggregation_func}'")
            if aggregation_func == 'sum':
                result = joined.groupby(f'{boundary_id_column}_right')[value_column].sum().reset_index()
            elif aggregation_func == 'mean':
                result = joined.groupby(f'{boundary_id_column}_right')[value_column].mean().reset_index()
            elif aggregation_func == 'std':
                result = joined.groupby(f'{boundary_id_column}_right')[value_column].std().reset_index()
            result = result.rename(columns={f'{boundary_id_column}_right': boundary_id_column})
        else:
            raise ValueError(f"Unsupported aggregation function: {aggregation_func}")
        
        # Merge back with boundary geometry
        boundary_result = boundaries.merge(result, on=boundary_id_column, how='left')
        boundary_result = boundary_result.fillna(0)
        
        logger.info(f"Aggregated data for {len(boundary_result)} boundaries")
        return boundary_result
    
    def create_density_surface(
        self, 
        data: gpd.GeoDataFrame, 
        bandwidth: float,
        grid_size: Optional[float] = None,
        kernel: str = 'gaussian'
    ) -> gpd.GeoDataFrame:
        """
        Create a density surface from point data using kernel density estimation.
        
        Args:
            data: Input GeoDataFrame with point geometries
            bandwidth: Bandwidth for kernel density estimation
            grid_size: Size of output grid cells (defaults to bandwidth/4)
            kernel: Kernel type ('gaussian', 'uniform', 'triangular')
            
        Returns:
            GeoDataFrame with density surface grid
        """
        logger.info(f"Creating density surface for {len(data)} points")
        
        if grid_size is None:
            grid_size = bandwidth / 4
        
        # Extract point coordinates
        if not all(data.geometry.geom_type == 'Point'):
            # Convert to centroids if not points
            data = data.copy()
            data.geometry = data.geometry.centroid
        
        coords = np.array([[geom.x, geom.y] for geom in data.geometry])
        
        # Create grid for density estimation
        bounds = data.total_bounds
        grid = self._create_grid(bounds, grid_size)
        
        # Calculate grid centroids
        grid_coords = np.array([[geom.centroid.x, geom.centroid.y] for geom in grid.geometry])
        
        # Compute density using vectorized operations
        if self.use_gpu and len(coords) > 1000:
            density_values = self._compute_density_gpu(coords, grid_coords, bandwidth, kernel)
        else:
            density_values = self._compute_density_cpu(coords, grid_coords, bandwidth, kernel)
        
        # Add density values to grid
        grid['density'] = density_values
        
        logger.info(f"Created density surface with {len(grid)} cells")
        return grid
    
    def _create_grid(self, bounds: Tuple[float, float, float, float], grid_size: float) -> gpd.GeoDataFrame:
        """Create a regular grid covering the given bounds."""
        minx, miny, maxx, maxy = bounds
        
        # Create grid coordinates
        x_coords = np.arange(minx, maxx + grid_size, grid_size)
        y_coords = np.arange(miny, maxy + grid_size, grid_size)
        
        # Create grid cells
        grid_cells = []
        grid_ids = []
        
        for i, x in enumerate(x_coords[:-1]):
            for j, y in enumerate(y_coords[:-1]):
                cell = box(x, y, x + grid_size, y + grid_size)
                grid_cells.append(cell)
                grid_ids.append(f"grid_{i}_{j}")
        
        return gpd.GeoDataFrame({
            'grid_id': grid_ids,
            'geometry': grid_cells
        })
    
    def _aggregate_by_grid_chunked(
        self, 
        data: gpd.GeoDataFrame, 
        grid: gpd.GeoDataFrame,
        aggregation_func: str,
        value_column: Optional[str]
    ) -> gpd.GeoDataFrame:
        """Process grid aggregation in chunks for large datasets."""
        logger.info(f"Processing grid aggregation in chunks of {self.chunk_size}")
        
        results = []
        for i in range(0, len(data), self.chunk_size):
            chunk = data.iloc[i:i + self.chunk_size]
            chunk_result = self.aggregate_by_grid(chunk, 0, aggregation_func, value_column)  # grid already created
            results.append(chunk_result)
        
        # Combine results
        if aggregation_func == 'count':
            combined = pd.concat([r[['grid_id', 'count']] for r in results])
            final_result = combined.groupby('grid_id')['count'].sum().reset_index()
        elif aggregation_func == 'sum':
            combined = pd.concat([r[['grid_id', value_column]] for r in results])
            final_result = combined.groupby('grid_id')[value_column].sum().reset_index()
        elif aggregation_func == 'mean':
            # For mean, we need to track counts and sums
            combined = pd.concat([r[['grid_id', value_column]] for r in results])
            final_result = combined.groupby('grid_id')[value_column].mean().reset_index()
        elif aggregation_func == 'std':
            combined = pd.concat([r[['grid_id', value_column]] for r in results])
            final_result = combined.groupby('grid_id')[value_column].std().reset_index()
        
        # Merge with grid geometry
        return grid.merge(final_result, on='grid_id', how='left').fillna(0)
    
    def _aggregate_by_boundaries_chunked(
        self,
        data: gpd.GeoDataFrame,
        boundaries: gpd.GeoDataFrame,
        aggregation_func: str,
        value_column: Optional[str],
        boundary_id_column: str
    ) -> gpd.GeoDataFrame:
        """Process boundary aggregation in chunks for large datasets."""
        logger.info(f"Processing boundary aggregation in chunks of {self.chunk_size}")
        
        results = []
        for i in range(0, len(data), self.chunk_size):
            chunk = data.iloc[i:i + self.chunk_size]
            chunk_result = self.aggregate_by_boundaries(
                chunk, boundaries, aggregation_func, value_column, boundary_id_column
            )
            results.append(chunk_result)
        
        # Combine results similar to grid aggregation
        if aggregation_func == 'count':
            combined = pd.concat([r[[boundary_id_column, 'count']] for r in results])
            final_result = combined.groupby(boundary_id_column)['count'].sum().reset_index()
        elif aggregation_func in ['sum', 'mean', 'std']:
            combined = pd.concat([r[[boundary_id_column, value_column]] for r in results])
            if aggregation_func == 'sum':
                final_result = combined.groupby(boundary_id_column)[value_column].sum().reset_index()
            elif aggregation_func == 'mean':
                final_result = combined.groupby(boundary_id_column)[value_column].mean().reset_index()
            elif aggregation_func == 'std':
                final_result = combined.groupby(boundary_id_column)[value_column].std().reset_index()
        
        return boundaries.merge(final_result, on=boundary_id_column, how='left').fillna(0)
    
    def _compute_density_cpu(
        self, 
        points: np.ndarray, 
        grid_coords: np.ndarray, 
        bandwidth: float, 
        kernel: str
    ) -> np.ndarray:
        """Compute kernel density estimation using CPU."""
        density = np.zeros(len(grid_coords))
        
        for i, grid_point in enumerate(grid_coords):
            distances = np.sqrt(np.sum((points - grid_point) ** 2, axis=1))
            
            if kernel == 'gaussian':
                weights = np.exp(-0.5 * (distances / bandwidth) ** 2)
            elif kernel == 'uniform':
                weights = (distances <= bandwidth).astype(float)
            elif kernel == 'triangular':
                weights = np.maximum(0, 1 - distances / bandwidth)
            else:
                raise ValueError(f"Unsupported kernel: {kernel}")
            
            density[i] = np.sum(weights)
        
        return density
    
    def _compute_density_gpu(
        self, 
        points: np.ndarray, 
        grid_coords: np.ndarray, 
        bandwidth: float, 
        kernel: str
    ) -> np.ndarray:
        """Compute kernel density estimation using GPU."""
        try:
            points_gpu = cp.asarray(points)
            grid_coords_gpu = cp.asarray(grid_coords)
            density_gpu = cp.zeros(len(grid_coords))
            
            for i in range(len(grid_coords)):
                distances = cp.sqrt(cp.sum((points_gpu - grid_coords_gpu[i]) ** 2, axis=1))
                
                if kernel == 'gaussian':
                    weights = cp.exp(-0.5 * (distances / bandwidth) ** 2)
                elif kernel == 'uniform':
                    weights = (distances <= bandwidth).astype(float)
                elif kernel == 'triangular':
                    weights = cp.maximum(0, 1 - distances / bandwidth)
                
                density_gpu[i] = cp.sum(weights)
            
            return cp.asnumpy(density_gpu)
        except Exception as e:
            logger.warning(f"GPU density computation failed: {e}, falling back to CPU")
            return self._compute_density_cpu(points, grid_coords, bandwidth, kernel)


class SpatialSampler:
    """
    Handles sampling strategies for large datasets with support for different
    sampling methods and adaptive strategies.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the SpatialSampler.
        
        Args:
            random_seed: Random seed for reproducible sampling
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        logger.info(f"SpatialSampler initialized with random seed {random_seed}")
    
    def systematic_sample(
        self, 
        bounds: Tuple[float, float, float, float], 
        sample_size: int
    ) -> List[Tuple[float, float]]:
        """
        Generate systematic sample points within given bounds.
        
        Args:
            bounds: Bounding box (minx, miny, maxx, maxy)
            sample_size: Number of sample points to generate
            
        Returns:
            List of (x, y) coordinate tuples
        """
        logger.info(f"Generating {sample_size} systematic sample points")
        
        minx, miny, maxx, maxy = bounds
        
        # Calculate grid dimensions for systematic sampling
        aspect_ratio = (maxx - minx) / (maxy - miny)
        n_cols = int(np.sqrt(sample_size * aspect_ratio))
        n_rows = int(sample_size / n_cols)
        
        # Adjust to get closer to desired sample size
        if n_cols * n_rows < sample_size:
            n_rows += 1
        
        # Generate systematic grid
        x_step = (maxx - minx) / n_cols
        y_step = (maxy - miny) / n_rows
        
        sample_points = []
        for i in range(n_cols):
            for j in range(n_rows):
                if len(sample_points) >= sample_size:
                    break
                x = minx + (i + 0.5) * x_step
                y = miny + (j + 0.5) * y_step
                sample_points.append((x, y))
        
        logger.info(f"Generated {len(sample_points)} systematic sample points")
        return sample_points[:sample_size]
    
    def stratified_sample(
        self, 
        regions: gpd.GeoDataFrame, 
        samples_per_region: int,
        region_id_column: str = 'id'
    ) -> List[Tuple[float, float, str]]:
        """
        Generate stratified sample points within different regions.
        
        Args:
            regions: GeoDataFrame with region polygons
            samples_per_region: Number of samples per region
            region_id_column: Column name for region identifiers
            
        Returns:
            List of (x, y, region_id) tuples
        """
        logger.info(f"Generating {samples_per_region} stratified samples per region")
        
        sample_points = []
        
        for idx, region in regions.iterrows():
            region_id = region[region_id_column] if region_id_column in regions.columns else idx
            bounds = region.geometry.bounds
            
            # Generate random points within region bounds
            region_samples = []
            attempts = 0
            max_attempts = samples_per_region * 10  # Avoid infinite loops
            
            while len(region_samples) < samples_per_region and attempts < max_attempts:
                x = np.random.uniform(bounds[0], bounds[2])
                y = np.random.uniform(bounds[1], bounds[3])
                point = Point(x, y)
                
                if region.geometry.contains(point):
                    region_samples.append((x, y, region_id))
                
                attempts += 1
            
            sample_points.extend(region_samples)
        
        logger.info(f"Generated {len(sample_points)} stratified sample points")
        return sample_points
    
    def adaptive_sample(
        self, 
        initial_results: Dict[str, Any], 
        target_precision: float,
        bounds: Tuple[float, float, float, float],
        max_iterations: int = 5
    ) -> List[Tuple[float, float]]:
        """
        Generate adaptive sample points based on initial analysis results.
        
        Args:
            initial_results: Results from initial sampling with variance estimates
            target_precision: Target coefficient of variation
            bounds: Bounding box for sampling
            max_iterations: Maximum number of adaptive iterations
            
        Returns:
            List of (x, y) coordinate tuples for additional sampling
        """
        logger.info(f"Generating adaptive samples targeting precision {target_precision}")
        
        # Extract variance information from initial results
        if 'variance' not in initial_results or 'mean' not in initial_results:
            logger.warning("Insufficient variance information for adaptive sampling")
            return self.systematic_sample(bounds, 100)  # Fallback
        
        variance = initial_results['variance']
        mean = initial_results['mean']
        current_n = initial_results.get('sample_size', 100)
        
        # Calculate coefficient of variation
        cv = np.sqrt(variance) / mean if mean > 0 else float('inf')
        
        if cv <= target_precision:
            logger.info(f"Target precision already achieved (CV: {cv:.4f})")
            return []
        
        # Calculate additional samples needed
        # Using formula: n_new = n_current * (CV_current / CV_target)^2
        n_additional = int(current_n * (cv / target_precision) ** 2 - current_n)
        n_additional = min(n_additional, 10000)  # Cap at reasonable limit
        
        logger.info(f"Adding {n_additional} adaptive samples (current CV: {cv:.4f})")
        
        # Generate additional systematic samples
        # In a more sophisticated implementation, this could focus on high-variance areas
        return self.systematic_sample(bounds, n_additional)
    
    def random_sample(
        self, 
        bounds: Tuple[float, float, float, float], 
        sample_size: int
    ) -> List[Tuple[float, float]]:
        """
        Generate random sample points within given bounds.
        
        Args:
            bounds: Bounding box (minx, miny, maxx, maxy)
            sample_size: Number of sample points to generate
            
        Returns:
            List of (x, y) coordinate tuples
        """
        logger.info(f"Generating {sample_size} random sample points")
        
        minx, miny, maxx, maxy = bounds
        
        x_coords = np.random.uniform(minx, maxx, sample_size)
        y_coords = np.random.uniform(miny, maxy, sample_size)
        
        sample_points = list(zip(x_coords, y_coords))
        
        logger.info(f"Generated {len(sample_points)} random sample points")
        return sample_points
    
    def create_sample_polygons(
        self, 
        sample_points: List[Tuple[float, float]], 
        buffer_size: float
    ) -> gpd.GeoDataFrame:
        """
        Create buffer polygons around sample points for data extraction.
        
        Args:
            sample_points: List of (x, y) coordinate tuples
            buffer_size: Buffer radius around each point
            
        Returns:
            GeoDataFrame with sample polygons
        """
        logger.info(f"Creating {len(sample_points)} sample polygons with buffer {buffer_size}")
        
        geometries = []
        sample_ids = []
        
        for i, (x, y) in enumerate(sample_points):
            point = Point(x, y)
            polygon = point.buffer(buffer_size)
            geometries.append(polygon)
            sample_ids.append(f"sample_{i}")
        
        return gpd.GeoDataFrame({
            'sample_id': sample_ids,
            'geometry': geometries
        })


# Utility functions for chunked processing
def process_in_chunks(
    data: Union[pd.DataFrame, gpd.GeoDataFrame],
    processing_func: callable,
    chunk_size: int = 10000,
    **kwargs
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Process large datasets in chunks to manage memory usage.
    
    Args:
        data: Input DataFrame or GeoDataFrame
        processing_func: Function to apply to each chunk
        chunk_size: Size of each chunk
        **kwargs: Additional arguments for processing function
        
    Returns:
        Processed DataFrame or GeoDataFrame
    """
    logger.info(f"Processing {len(data)} records in chunks of {chunk_size}")
    
    results = []
    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i + chunk_size]
        chunk_result = processing_func(chunk, **kwargs)
        results.append(chunk_result)
    
    # Combine results
    if isinstance(data, gpd.GeoDataFrame):
        return gpd.GeoDataFrame(pd.concat(results, ignore_index=True))
    else:
        return pd.concat(results, ignore_index=True)


def estimate_memory_usage(data: Union[pd.DataFrame, gpd.GeoDataFrame]) -> float:
    """
    Estimate memory usage of a DataFrame in MB.
    
    Args:
        data: Input DataFrame or GeoDataFrame
        
    Returns:
        Estimated memory usage in MB
    """
    memory_usage = data.memory_usage(deep=True).sum()
    return memory_usage / (1024 * 1024)  # Convert to MB