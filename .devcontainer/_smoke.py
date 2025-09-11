import cupy, cudf, cuspatial, geopandas, rasterio, fiona, pyproj, pyarrow
print("GPU:", cupy.cuda.runtime.getDeviceCount())
print("cudf", cudf.__version__)
print("cuspatial", cuspatial.__version__)
print("geopandas", geopandas.__version__)
print("rasterio GDAL", getattr(rasterio, "__gdal_version__", "?"))
print("pyproj", pyproj.__version__)
print("pyarrow", pyarrow.__version__)
