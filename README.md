# GIS + RAPIDS Dev Container

A reproducible geospatial data science environment with CUDA-accelerated RAPIDS, built on micromamba. Open the folder in a VS Code Dev Container to get Python, GPU libraries, Jupyter, and GitHub tooling preconfigured.

## What’s Inside

- CUDA-enabled base image (CUDA 12.2) with micromamba environment `gis`.
- RAPIDS 24.12 (cuDF, cuSpatial) and core geo stack (GeoPandas, Shapely, Rasterio, GDAL).
- JupyterLab, IPyKernel, and VS Code Jupyter integration.
- Git + GitHub CLI (`gh`) + SSH agent forwarding.
- Data bind mount of `./data` to `/workspace/data` inside the container.

Key files:
- `.devcontainer/devcontainer.json` – Dev container definition and features
- `.devcontainer/Dockerfile` – Base image and environment creation
- `.devcontainer/environment.yml` – Conda environment spec (`gis`)
- `.devcontainer/postCreate.sh` – Post-create checks and kernel registration
- `.devcontainer/GITHUB_AUTH.md` – Safe GitHub sign-in options
- `code/overture_analysis/` – Example notebooks

## Prerequisites

- Docker Desktop (Linux, macOS, or Windows + WSL2)
- VS Code with “Dev Containers” extension
- NVIDIA GPU (optional) + current NVIDIA driver
  - Linux: NVIDIA Container Toolkit
  - Windows: WSL2 + GPU support in Docker

## Getting Started

1) Open in Dev Container
- In VS Code: “Dev Containers: Reopen in Container” (or use the green corner button).
- The container builds the `gis` environment and registers a Jupyter kernel.

2) Jupyter usage
- Use VS Code notebooks or open a terminal and run:
  ```bash
  micromamba activate gis
  jupyter lab --ip 0.0.0.0 --port 8888 --no-browser
  ```
- Default port 8888 is forwarded. Select the kernel “Python (gis)”.

3) Data
- Place files under `./data` on the host. They appear in the container at `/workspace/data`.
- To mount an external host directory without breaking portability, create a local override file `.devcontainer/devcontainer.local.json` (not checked in) with one of the following:
  - Using an environment variable (recommended):
    `{ "mounts": [ "source=${localEnv:GIS_DATA_DIR},target=${containerWorkspaceFolder}/gis_data,type=bind,rw" ] }`
    Then set `GIS_DATA_DIR` on your host to the desired path (e.g., `D:\\data`).
  - Hardcoding a path (Windows example):
    `{ "mounts": [ "source=c:\\\\data,target=${containerWorkspaceFolder}/gis_data,type=bind,rw" ] }`
  Ensure the path exists and (on Docker Desktop) the drive is shared under Settings > Resources > File sharing. If the path isn’t accessible, remove or fix the local override.

4) GPU check (optional)
- After the container starts, the post-create script prints versions and a CUDA availability line.
- From a terminal:
  ```python
  import cupy; print(cupy.cuda.runtime.getDeviceCount() > 0)
  ```
  True indicates the container can access your GPU.

## GitHub Authentication

This project includes Git and GitHub CLI with safe, consistent sign-in methods.
- Recommended: GitHub CLI over HTTPS
  ```bash
  gh auth login --hostname github.com --git-protocol https --web
  gh auth setup-git
  gh auth status
  ```
- SSH with agent forwarding (if you already use SSH keys on the host)
- Env token passthrough for CI/advanced users (`GH_TOKEN`/`GITHUB_TOKEN`)

See detailed options and security tips in `.devcontainer/GITHUB_AUTH.md`.

## Common Tasks

- Activate environment in a shell:
  ```bash
  micromamba activate gis
  ```
- Run a quick check of RAPIDS/geo stack:
  ```bash
  python - <<'PY'
  import cupy, cudf, cuspatial, geopandas, rasterio
  print('CUDA devices > 0:', cupy.cuda.runtime.getDeviceCount() > 0)
  print('cuDF:', cudf.__version__, 'cuSpatial:', cuspatial.__version__)
  print('GeoPandas:', geopandas.__version__, 'GDAL via rasterio:', getattr(rasterio, '__gdal_version__', '?'))
  PY
  ```
- Work with notebooks in `code/overture_analysis/` (e.g., `intro.ipynb`).

## Troubleshooting

- No GPU detected
  - Ensure host NVIDIA driver is installed and Docker has GPU access.
  - On Windows, enable WSL2 integration and GPU support in Docker.
  - Rebuild the container after driver/toolkit changes.
- Build failures on geospatial libs
  - Rebuild without cache; ensure Docker has sufficient RAM.
- Credential issues with GitHub
  - Run `gh auth login` and `gh auth setup-git`; verify with `gh auth status`.

## Notes

 - Workspace path inside container: `/workspace`.
- Python path in VS Code: `/opt/conda/envs/gis/bin/python`.
- Port 8888 labeled as “Jupyter” in the Dev Container config.
