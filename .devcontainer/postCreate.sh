#!/usr/bin/env bash
set -euo pipefail
eval "$(micromamba shell hook -s bash)"
micromamba activate gis

python -m ipykernel install --user --name=gis --display-name "Python (gis)"

python - <<'PY'
import cupy, cudf, cuspatial, pyarrow, geopandas, shapely
import rasterio, fiona, pyproj
import numba, numpy

print("CUDA available via CuPy:", cupy.cuda.runtime.getDeviceCount() > 0)
print("cuDF version:", cudf.__version__)
print("cuSpatial version:", cuspatial.__version__)
print("PyArrow:", pyarrow.__version__)
print("GeoPandas:", geopandas.__version__)
print("Shapely:", shapely.__version__)
print("GDAL (rasterio):", getattr(rasterio, "__gdal_version__", "?"))
PY

python - <<'PY'
import duckdb, polars, pyarrow as pa
print("DuckDB:", duckdb.__version__)
print("Polars:", polars.__version__)
print("PyArrow:", pa.__version__)
con = duckdb.connect()
print(con.sql("select 1 as ok").fetchall())
PY
