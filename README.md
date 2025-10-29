# GIS Dev Containers

A reproducible geospatial data science environment built on micromamba. Two Dev Container variants let you choose between a CPU-only stack or a CUDA-accelerated RAPIDS stack whenever you reopen the repository.

## Container Variants
- **CPU** (`.devcontainer/cpu/devcontainer.json`) – Uses the Ubuntu Jammy micromamba image and installs a pure CPU geospatial toolkit (GeoPandas, GDAL, Rasterio, Cartopy, plus the viz stack and DuckDB/Polars).
- **GPU** (`.devcontainer/gpu/devcontainer.json`) – Builds from the CUDA 12.2 micromamba image, adds RAPIDS 24.12 (cuDF, cuSpatial), CuPy, and CUDA helpers, and enables the NVIDIA container toolkit with `--gpus=all`.

Both variants inherit shared settings (mounts, VS Code extensions, environment variables, port forwarding, GitHub token passthrough, etc.) from `.devcontainer/base/devcontainer.json`.

## What's Inside
- Conda environment `gis` with JupyterLab, IPyKernel, GitHub CLI, and the core Python geo libraries.
- Post-create script that registers the kernel and reports which optional GPU packages were detected.
- Data bind mount of `./data` on the host to `/workspace/data` in the container (plus any overrides you add in each variant’s `devcontainer.local.json`).
- GPU variant only: RAPIDS, CuPy, CUDA runtime environment, RMM/UCX tuning.

Key files:
- `.devcontainer/base/devcontainer.json` – Shared container settings.
- `.devcontainer/cpu/Dockerfile` and `environment.yml` – CPU build context.
- `.devcontainer/gpu/Dockerfile` and `environment.yml` – GPU build context.
- `.devcontainer/cpu/devcontainer.local.json` / `.devcontainer/gpu/devcontainer.local.json` – Per-variant overrides (edit locally for custom mounts, env vars).
- `.devcontainer/postCreate.sh` – Post-create automation (GPU-aware).
- `.devcontainer/GITHUB_AUTH.md` – GitHub authentication guidance.

## Prerequisites
- Docker Desktop (Linux, macOS, or Windows with WSL2).
- VS Code with the Dev Containers extension (or GitHub Codespaces).
- For GPU work: NVIDIA driver plus the NVIDIA Container Toolkit (Linux) or WSL2 GPU support (Windows).

## Getting Started
1. When prompted to reopen in a container, pick **gis_conda_cpu** or **gis_conda_gpu**. You can switch later via “Dev Containers: Reopen in Container…” and choosing the other profile.
2. After the container builds, activate the environment as needed:
   ```bash
   micromamba activate gis
   ```
3. Launch Jupyter if you prefer a browser session:
   ```bash
   jupyter lab --ip 0.0.0.0 --port 8888 --no-browser
   ```
4. Drop host data into `./data` or customize the variant’s `devcontainer.local.json` to add extra mounts. Example:
   ```json
   {
     "mounts": [
       "source=${localEnv:GIS_DATA_DIR},target=${containerWorkspaceFolder}/gis_data,type=bind,rw"
     ]
   }
   ```

## GPU Quick Check
Inside the GPU container run:
```python
import cupy, cudf, cuspatial
print("CUDA devices:", cupy.cuda.runtime.getDeviceCount())
print("cuDF:", cudf.__version__, "cuSpatial:", cuspatial.__version__)
```
The post-create script also prints availability if those packages are present.

## GitHub Authentication
Git and GitHub CLI are preinstalled. Recommended flow:
```bash
gh auth login --hostname github.com --git-protocol https --web
gh auth setup-git
gh auth status
```
Token passthrough (`GH_TOKEN`/`GITHUB_TOKEN`) is supported, and SSH agent forwarding works if you enable it on the host. See `.devcontainer/GITHUB_AUTH.md` for details.

## Common Tasks
- Run the optional smoke check: `.devcontainer/_smoke.py` exercises the GPU stack (only works if RAPIDS is installed).
- Explore notebooks in `code/overture_analysis/`.
- Confirm GitHub CLI status: `gh auth status`.

## Troubleshooting
- **No GPU detected:** Verify host driver/toolkit installation and rebuild the GPU container without cache.
- **Conda solve issues:** Rebuild the container; the CPU and GPU environments are separate, so confirm you opened the intended profile.
- **Data mount missing:** Ensure the host path exists and, on Docker Desktop, that the drive is shared.
