#!/usr/bin/env bash
set -euo pipefail
eval "$(micromamba shell hook -s bash)"
micromamba activate gis

python -m ipykernel install --user --name=gis --display-name "Python (gis)"

python - <<'PY'
from importlib import import_module, util

def maybe_import(name):
    if util.find_spec(name) is None:
        print(f"{name}: not installed")
        return None
    module = import_module(name)
    version = getattr(module, "__version__", "unknown")
    print(f"{name}: {version}")
    return module

cupy = maybe_import("cupy")
if cupy:
    try:
        print("CUDA available via CuPy:", cupy.cuda.runtime.getDeviceCount() > 0)
    except Exception as exc:  # pragma: no cover
        print("CUDA check failed:", exc)

for mod in ("cudf", "cuspatial", "geopandas", "shapely", "rasterio", "fiona", "pyproj", "pyarrow", "numpy", "numba"):
    module = maybe_import(mod)
    if mod == "rasterio" and module is not None:
        print("GDAL (rasterio):", getattr(module, "__gdal_version__", "?"))
PY

python - <<'PY'
from importlib import import_module, util

def maybe_import(name):
    if util.find_spec(name) is None:
        print(f"{name}: not installed")
        return None
    module = import_module(name)
    version = getattr(module, "__version__", "unknown")
    print(f"{name}: {version}")
    return module

duckdb = maybe_import("duckdb")
polars = maybe_import("polars")
pa = maybe_import("pyarrow")
if duckdb:
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
