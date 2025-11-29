# Commands Log

- **Sample building sources**  
  Generate a 10k sample of building features from the 2025-10-22.0 release to inspect source distribution; writes a JSON report (multi-source Parquet output disabled).  
  ```
  /opt/conda/envs/gis/bin/python code/overture_analysis/data_processing/sample_building_sources.py \
  --input-dir /workspaces/micromamba_cuda/gis_data/overturemaps-us-west-2/release/2025-10-22.0/theme=buildings/type=building \
  --sample-size 10000 \
  --output /workspaces/micromamba_cuda/data/results/buildings_sources_report_10k.json
  ```

- **Sample building sources with GeoJSON examples**  
  Same as above but also writes up to 100 sample buildings per source combination to GeoJSON for quick map inspection (multi-source Parquet output disabled).  
  ```
  /opt/conda/envs/gis/bin/python code/overture_analysis/data_processing/sample_building_sources.py \
  --input-dir /workspaces/micromamba_cuda/gis_data/overturemaps-us-west-2/release/2025-10-22.0/theme=buildings/type=building \
  --sample-size 1000 \
  --output /workspaces/micromamba_cuda/data/results/buildings_sources_report_1k.json \
  --samples-geojson /workspaces/micromamba_cuda/data/results/building_samples_1k.geojson
  ```

- **Summarize buildings by source (test run)**  
  Smoke-test the summarization pipeline on a small test dataset to verify arguments and output structure before the full run; writes summaries and a log JSON under `data/results/buildings_source_summary_test`.  
  ```
  /opt/conda/envs/gis/bin/python code/overture_analysis/data_processing/summarize_buildings_by_source.py \
  /workspaces/micromamba_cuda/gis_data/overturemaps-us-west-2/test_buildings \
  --results-dir data/results/buildings_source_summary_test \
  --log-json data/results/buildings_source_summary_test/log.json \
  --max-workers 8
  ```

- **Summarize buildings by source (full run)**  
  Run the full summarization over the 2025-10-22.0 building release to produce aggregated source stats and logs in `data/results/buildings_source_summary`.  
  ```
  /opt/conda/envs/gis/bin/python code/overture_analysis/data_processing/summarize_buildings_by_source.py \
  /workspaces/micromamba_cuda/gis_data/overturemaps-us-west-2/release/2025-10-22.0/theme=buildings/type=building \
  --results-dir data/results/buildings_source_summary \
  --log-json data/results/buildings_source_summary/log.json \
  --max-workers 8
  ```

- **Fix metadata (test)**  
  Validate `fix_metadata.py` on the test summaries to ensure metadata and CRS corrections work before applying to the full dataset; keeps operations limited to the test results directory.  - not works!
  ```
  /opt/conda/envs/gis/bin/python code/overture_analysis/data_processing/fix_metadata.py \
  --summary-dir data/results/buildings_source_summary_test \
  --input-dir /workspaces/micromamba_cuda/gis_data/overturemaps-us-west-2/test_buildings \
  --crs EPSG:4326 \
  --max-workers 8
  ```

- **Aggregate region dataset counts (full run, optimized for 20 cores)**  
  Count buildings by dataset per region using the summary Parquets (s1 only). Keeps all region columns; drops only helper `_bbox_*` and `region_idx`. Throughput benchmark on the first file: ~1k rows/sec per worker at `--chunk-size 400000`; with ~2.54B rows total this projects to ~2.3 days on 12 workers (20-core host leaves headroom).  
  ```
  /opt/conda/envs/gis/bin/python code/overture_analysis/data_processing/aggregate_region_dataset_counts_duckdb.py \
  data/results/buildings_source_summary \
  data/results/region_area-2025-10-22.0.parquet \
  --max-workers 12 \
  --threads-per-worker 8 \
  --chunk-size 400000
  ```
