# Project Structure

## Root Directory Layout
```
├── .devcontainer/          # Dev container configuration
├── .kiro/                  # Kiro AI assistant configuration
├── code/                   # Analysis notebooks and scripts
├── data/                   # Local data files (mounted to container)
├── gis_data/               # External data mount point
└── README.md               # Project documentation
```

## Key Directories

### `.devcontainer/`
Container configuration and setup scripts:
- `devcontainer.json` - VS Code dev container definition
- `Dockerfile` - Container image specification
- `environment.yml` - Conda environment dependencies
- `postCreate.sh` - Post-container creation setup
- `GITHUB_AUTH.md` - GitHub authentication guide

### `code/`
Analysis code organized by project:
- `demo/` - Basic demonstration notebooks
- `overture_analysis/` - Overture Maps data analysis
  - `functions/` - Reusable analysis functions
  - `*.ipynb` - Jupyter notebooks for specific analyses

### `data/`
Local data storage (bind mounted to container):
- `OSM Projects/` - OpenStreetMap related data and documentation
  - `Code/`, `Data/`, `Papers/`, `Proposals/` - Organized by type
- `results/` - Analysis outputs and processed data
- `.gitkeep` - Ensures directory exists in git

### `gis_data/`
External data mount point (configurable in devcontainer.json)

## File Naming Conventions

### Notebooks
- Use descriptive names: `learn_buildings.ipynb`, `investigate_buildings_sources.ipynb`
- Prefix with action: `learn_`, `investigate_`, `analyze_`
- Use underscores for separation

### Data Files
- Results use descriptive names with format: `buildings_investigation_result.parquet`
- Logs include `_log` suffix: `buildings_investigation_log.csv`
- Use appropriate extensions: `.parquet` for large datasets, `.csv` for logs

## Data Organization Principles
- **Raw data**: Store in `data/` subdirectories by source/project
- **Processed data**: Save results in `data/results/`
- **External data**: Mount large datasets to `gis_data/` via container config
- **Code separation**: Keep analysis notebooks in `code/` organized by project theme

## Container Paths
- Host `./data` → Container `/workspace/data`
- Host `./gis_data` → Container `/workspace/gis_data` (if configured)
- Working directory: `/workspace` (project root)

## Development Workflow
1. Place notebooks in appropriate `code/` subdirectory
2. Store input data in `data/` with logical organization
3. Save analysis outputs to `data/results/`
4. Use relative paths from `/workspace` in notebooks
5. Activate `gis` environment before running analysis