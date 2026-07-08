# Scene Validation Repository

A repository for comparing simulated and real multispectral scene data

## Structure

- `src/`: shared reusable helper functions
- `scripts/`: main executable scripts 
- `notebooks/`: exploratory analysis and figure development
- `results/`: generated plots and tables on local machine


## Main workflow

1. Run `scripts/Simulated_Real_data.py` to generate `data.pkl` in the script directory.
2. Run `scripts/Val_with_bands_3years.py` for three-year band comparisons.
3. Run `scripts/Val_with_bands_1year.py` for one-year band comparisons.
4. Run `scripts/Val_with_indices.py` for vegetation index validation.
5. Run `scripts/validation_demo.py` for vegetation index demo validation.
6. Use `notebooks/exploration_indices.ipynb` for exploratory figure refinement.

## Main files

### `scripts/Simulated_Real_data.py`
Upstream script that builds `data.pkl` from simulated scene ROIs and real clipped raster bands after outlier removal and NaN masking.

### `scripts/Val_with_bands_3years.py`
Three-year reflectance band comparison workflow.

### `scripts/Val_with_bands_1year.py`
One-year reflectance band comparison workflow.

### `scripts/Val_with_indices.py`
Index computation, KS testing, summary export, and Shannon-vs-metric plotting.

### `src/data_preparation.py`
Reusable ENVI/raster utilities and outlier helpers.

### `src/indices_and_metrics.py`
Reusable vegetation index and statistics helpers.

### `src/validation_stats.py`
Reusable validation metrics.

### `src/validation_summary.py`
stats/helper module.

### `src/figures_indices.py`
plotting helper module.


