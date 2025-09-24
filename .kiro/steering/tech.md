# Technology Stack

## Development Environment
- **Container**: Docker Dev Container with VS Code integration
- **Base**: CUDA 12.2 enabled image with micromamba
- **Python**: 3.12 in conda environment named `gis`
- **Package Manager**: micromamba (faster conda alternative)

## Core Libraries
### GPU Acceleration
- **RAPIDS 24.12**: cuDF, cuSpatial for GPU-accelerated data processing
- **CuPy**: GPU array library
- **CUDA 12.2**: GPU compute platform

### Geospatial Stack
- **GeoPandas**: Spatial data manipulation
- **Shapely 2.0.4**: Geometric operations
- **GDAL/Rasterio**: Raster data I/O
- **Fiona**: Vector data I/O
- **Cartopy**: Cartographic projections
- **RioXarray**: Raster data with xarray

### Data Processing
- **Pandas 2.2.2**: Tabular data manipulation
- **NumPy 1.26.4**: Numerical computing
- **PyArrow**: Columnar data format
- **DuckDB 1.1.3**: In-process OLAP database
- **Polars**: Fast DataFrame library

### Visualization & Analysis
- **JupyterLab 4.2.5**: Interactive notebooks
- **Matplotlib 3.8.4**: Plotting
- **Scikit-learn 1.4.2**: Machine learning
- **Leafmap, KeplerGL, PyDeck**: Interactive mapping

## Common Commands

### Environment Management
```bash
# Activate the gis environment
micromamba activate gis

# Check GPU availability
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount() > 0)"
```

### Development Workflow
```bash
# Start Jupyter Lab
jupyter lab --ip 0.0.0.0 --port 8888 --no-browser

# Quick stack verification
python - <<'PY'
import cupy, cudf, cuspatial, geopandas, rasterio
print('CUDA devices > 0:', cupy.cuda.runtime.getDeviceCount() > 0)
print('cuDF:', cudf.__version__, 'cuSpatial:', cuspatial.__version__)
print('GeoPandas:', geopandas.__version__, 'GDAL via rasterio:', getattr(rasterio, '__gdal_version__', '?'))
PY
```

### GitHub Authentication
```bash
# Web-based login (recommended)
gh auth login --hostname github.com --git-protocol https --web
gh auth setup-git
gh auth status
```

## Build System
- **Docker**: Container build and runtime
- **micromamba**: Package installation and environment management
- **VS Code Dev Containers**: Development environment orchestration

## Key Configurations
- Python interpreter: `/opt/conda/envs/gis/bin/python`
- Jupyter port: 8888 (auto-forwarded)
- Workspace mount: `/workspace` in container
- Data mount: `./data` → `/workspace/data`