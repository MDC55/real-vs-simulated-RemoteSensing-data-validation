
# ANG hyperspectral validation repo

This repo keeps the workflow close to the original scripts, but moves repeated helper logic into `src/` and uses repo-style paths.

## Structure
- `src/`: shared helpers
- `scripts/`: main preprocessing and validation entry scripts
- `notebooks/`: exploratory notebooks
- `data/`: optional raw and processed data locations
- `results/`: figures and tables

## Main scripts
- `scripts/build_real_data_pickle.py`
- `scripts/build_simulated_data_pickle.py`
- `scripts/validate_real_vs_sim_ANG_using_narrowband_indices`
- `scripts/validate_real_vs_sim_ANG_using_uas_indices.py`

