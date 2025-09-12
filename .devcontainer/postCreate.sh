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

# --- GitHub auth (token-based) ---
# If GH_TOKEN or GITHUB_TOKEN is provided from host, log in and set up git.
if [[ -n "${GH_TOKEN:-}" || -n "${GITHUB_TOKEN:-}" ]]; then
  TOKEN="${GH_TOKEN:-${GITHUB_TOKEN:-}}"
  if command -v gh >/dev/null 2>&1; then
    echo "Configuring GitHub CLI authentication..."
    # Avoid interactive prompts; ignore error if already logged in
    printf '%s' "$TOKEN" | gh auth login --hostname github.com --with-token || true
    gh auth setup-git || true
  else
    echo "Warning: gh CLI not found in PATH; skipping GitHub auth setup" >&2
  fi
fi

echo "--- GitHub CLI status (if installed) ---" || true
if command -v gh >/dev/null 2>&1; then
  gh --version || true
  gh auth status || true
  echo "To sign in run: gh auth login --hostname github.com --git-protocol https --web" || true
  echo "(Headless? Use: gh auth login --hostname github.com --git-protocol https --device)" || true
  echo "Then run: gh auth setup-git" || true
fi
