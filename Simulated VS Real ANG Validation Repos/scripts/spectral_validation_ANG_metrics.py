"""
Quantitative spectral validation: Real vs Simulated AVIRIS-NG hyperspectral reflectance
=========================================================================================
Written in response to reviewer comment:
    "The hyperspectral comparison still relies mainly on visual comparison of
    mean spectra, variability envelopes, vegetation-index means, and percentage
    differences. Please add at least one direct spectral similarity or error
    measure, such as the Spectral Angle Mapper, RMSE, normalized RMSE,
    correlation coefficient, or wavelength-wise error statistics."

This is fully standalone -- it does NOT import your `src.*` package, so it
will run in any environment with numpy / pandas / matplotlib. It reads
`realdata.pkl` (keys: 'data' [rows,cols,425 bands], 'wl' [425] in nm) and
`simulateddata.pkl` (keys: 'roi' [rows,cols,480 bands], 'wl_sim' [480] in nm)
directly.

WHY NOT A NAIVE PIXEL-WISE RMSE
--------------------------------
- The real cube (41x24 px here) and the simulated ROI (15x15 px here) are
  different spatial extents -- pixels are not spatially co-registered, so a
  pixel-to-pixel comparison is not meaningful.
- The two sensors/simulations were sampled on different wavelength grids
  (425 vs. 480 bands, ~5.0 nm vs. ~4.4 nm spacing, slightly different start
  wavelengths), so bands can't be compared by index -- they must be matched
  by wavelength.
- Both cubes contain severe noise/invalid values (including +-inf) inside
  the classic atmospheric water-vapor absorption windows (~1340-1460 nm and
  ~1790-1970 nm) and at the sensor edges (<400 nm, >2450 nm). These are
  excluded before computing any statistic, exactly as your existing
  `mask_noisy_wavelengths` step does.

WHAT THIS SCRIPT DOES
----------------------
1. Loads both cubes, builds a validity mask per dataset (edge cutoff +
   water-absorption windows), and reports how many bands were kept.
2. Wavelength-matches every valid real band to its nearest valid simulated
   band (within a small tolerance), giving a common set of paired
   wavelengths -- this is the standard way to compare hyperspectral data
   acquired/simulated on different band grids.
3. Computes, on the matched full spectrum (mean reflectance per band):
      - Spectral Angle Mapper (SAM, degrees) -- classic full-spectrum
        similarity metric.
      - RMSE and NRMSE (%) between mean spectra.
      - Pearson correlation coefficient (r) across ~300+ matched wavelengths.
4. Repeats RMSE/NRMSE/SAM/r per spectral region (VNIR / SWIR1 / SWIR2) as a
   compact "wavelength-wise error statistics" table.
5. Additionally computes a pooled, quantile-matched RMSE/NRMSE/r using the
   FULL pixel distributions (not just the mean), to capture spread as well
   as central tendency, addressing the same concern from a different angle.
6. Prints + saves compact CSV tables and a diagnostic plot.

Just drop this script in the same folder as realdata.pkl / simulateddata.pkl
and run it. Edit the CONFIG block below if your filenames/paths differ.
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Some bands (e.g. fully invalid/inf edge bands we mask out anyway) can be
# all-NaN in the raw cube; nanmean warns about this harmlessly.
warnings.filterwarnings('ignore', message='Mean of empty slice')

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
REAL_PKL = os.path.join(script_dir, 'realdata.pkl')
SIM_PKL = os.path.join(script_dir, 'simulateddata.pkl')
OUT_DIR = script_dir

WATER_BANDS_NM = [(1340, 1460), (1790, 2100)]   # classic strong water-vapor absorption windows
EDGE_RANGE_NM = (400, 2450)                      # drop noisy sensor edges outside this range
WAVELENGTH_MATCH_TOL_NM = 6.0                    # max allowed gap when pairing real<->sim bands
N_QUANTILES = 100                                # resolution for distribution (quantile) matching

REGIONS = {
    'VNIR (400-1000 nm)': (400, 1000),
    'SWIR1 (1000-1790 nm)': (1000, 1790),
    'SWIR2 (2100-2450 nm)': (2100, 2450),
}

# ---------------------------------------------------------------
# Load data
# ---------------------------------------------------------------
with open(REAL_PKL, 'rb') as f:
    real_pkl = pickle.load(f)
data_real = np.asarray(real_pkl['data'])     # (rows, cols, bands)
wl_real = np.asarray(real_pkl['wl'], dtype=float)

with open(SIM_PKL, 'rb') as f:
    sim_pkl = pickle.load(f)
data_sim = np.asarray(sim_pkl['roi'])        # (rows, cols, bands)
wl_sim = np.asarray(sim_pkl['wl_sim'], dtype=float)

print(f"Real cube:  {data_real.shape} (rows x cols x bands), wavelengths {wl_real.min():.1f}-{wl_real.max():.1f} nm")
print(f"Sim cube:   {data_sim.shape} (rows x cols x bands), wavelengths {wl_sim.min():.1f}-{wl_sim.max():.1f} nm")


# ---------------------------------------------------------------
# 1) Validity mask: drop sensor edges + water-absorption windows
# ---------------------------------------------------------------
def valid_wavelength_mask(wl):
    mask = (wl >= EDGE_RANGE_NM[0]) & (wl <= EDGE_RANGE_NM[1])
    for lo, hi in WATER_BANDS_NM:
        mask &= ~((wl >= lo) & (wl <= hi))
    return mask


mask_real = valid_wavelength_mask(wl_real)
mask_sim = valid_wavelength_mask(wl_sim)
print(f"\nKept {mask_real.sum()}/{len(wl_real)} real bands and {mask_sim.sum()}/{len(wl_sim)} sim bands "
      f"after removing edges + water-absorption windows.")

# replace any remaining non-finite values (rare) with NaN so they're excluded from means
data_real = np.where(np.isfinite(data_real), data_real, np.nan)
data_sim = np.where(np.isfinite(data_sim), data_sim, np.nan)

# ---------------------------------------------------------------
# 2) Wavelength-match: nearest valid sim band for every valid real band
# ---------------------------------------------------------------
idx_real_valid = np.where(mask_real)[0]
idx_sim_valid = np.where(mask_sim)[0]
wl_real_valid = wl_real[idx_real_valid]
wl_sim_valid = wl_sim[idx_sim_valid]

matched_real_idx, matched_sim_idx = [], []
for i, w in zip(idx_real_valid, wl_real_valid):
    j_local = np.argmin(np.abs(wl_sim_valid - w))
    if abs(wl_sim_valid[j_local] - w) <= WAVELENGTH_MATCH_TOL_NM:
        matched_real_idx.append(i)
        matched_sim_idx.append(idx_sim_valid[j_local])

matched_real_idx = np.array(matched_real_idx)
matched_sim_idx = np.array(matched_sim_idx)
matched_wl = wl_real[matched_real_idx]  # reference wavelength axis for the matched pairs
print(f"Matched {len(matched_real_idx)} wavelength pairs "
      f"({matched_wl.min():.1f}-{matched_wl.max():.1f} nm, tolerance {WAVELENGTH_MATCH_TOL_NM} nm).")


# ---------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------
def rmse(a, b):
    return np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


def nrmse(a, b, ref):
    rng = np.nanmax(ref) - np.nanmin(ref)
    return (rmse(a, b) / rng) * 100 if rng != 0 else np.nan


def pearson_r(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return np.nan
    return np.corrcoef(a, b)[0, 1]


def sam_degrees(v1, v2):
    v1, v2 = np.asarray(v1, dtype=float), np.asarray(v2, dtype=float)
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def quantile_match(real, sim, n_quantiles=N_QUANTILES):
    q = np.linspace(0, 1, n_quantiles)
    return np.nanquantile(real, q), np.nanquantile(sim, q)


# ---------------------------------------------------------------
# 3) Full-spectrum metrics on matched mean spectra
# ---------------------------------------------------------------
mean_real_full = np.nanmean(data_real, axis=(0, 1))
mean_sim_full = np.nanmean(data_sim, axis=(0, 1))

real_spec = mean_real_full[matched_real_idx]
sim_spec = mean_sim_full[matched_sim_idx]

full_spectrum_summary = {
    'N_matched_bands': len(matched_wl),
    'Wavelength_range_nm': f"{matched_wl.min():.0f}-{matched_wl.max():.0f}",
    'SAM_deg': round(sam_degrees(real_spec, sim_spec), 4),
    'RMSE': round(rmse(real_spec, sim_spec), 4),
    'NRMSE_%': round(nrmse(real_spec, sim_spec, real_spec), 4),
    'Pearson_r': round(pearson_r(real_spec, sim_spec), 4),
}

print("\n=== Full-spectrum metrics (mean reflectance, wavelength-matched bands) ===")
for k, v in full_spectrum_summary.items():
    print(f"{k}: {v}")

# ---------------------------------------------------------------
# 4) Per-region breakdown (compact "wavelength-wise" table)
# ---------------------------------------------------------------
region_rows = []
for region_name, (lo, hi) in REGIONS.items():
    in_region = (matched_wl >= lo) & (matched_wl <= hi)
    if in_region.sum() < 2:
        continue
    r_spec = real_spec[in_region]
    s_spec = sim_spec[in_region]
    region_rows.append({
        'Region': region_name,
        'N_bands': int(in_region.sum()),
        'SAM_deg': round(sam_degrees(r_spec, s_spec), 4),
        'RMSE': round(rmse(r_spec, s_spec), 4),
        'NRMSE_%': round(nrmse(r_spec, s_spec, r_spec), 4),
        'Pearson_r': round(pearson_r(r_spec, s_spec), 4),
    })

df_region = pd.DataFrame(region_rows)
print("\n=== Per-region error statistics (mean spectra) ===")
print(df_region.to_string(index=False))

# ---------------------------------------------------------------
# 5) Pooled, quantile-matched metrics using FULL pixel distributions
# ---------------------------------------------------------------
all_real_q, all_sim_q = [], []
for ri, si in zip(matched_real_idx, matched_sim_idx):
    real_band = data_real[:, :, ri].flatten()
    sim_band = data_sim[:, :, si].flatten()
    real_band = real_band[~np.isnan(real_band)]
    sim_band = sim_band[~np.isnan(sim_band)]
    if len(real_band) == 0 or len(sim_band) == 0:
        continue
    rq, sq = quantile_match(real_band, sim_band)
    all_real_q.append(rq)
    all_sim_q.append(sq)

all_real_q = np.concatenate(all_real_q)
all_sim_q = np.concatenate(all_sim_q)

pooled_distributional_summary = {
    'Pooled_RMSE_quantile_matched': round(rmse(all_real_q, all_sim_q), 4),
    'Pooled_NRMSE_%_quantile_matched': round(nrmse(all_real_q, all_sim_q, all_real_q), 4),
    'Pooled_Pearson_r_quantile_matched': round(pearson_r(all_real_q, all_sim_q), 4),
}

print("\n=== Pooled distributional summary (all matched bands, full pixel distributions) ===")
for k, v in pooled_distributional_summary.items():
    print(f"{k}: {v}")

# ---------------------------------------------------------------
# 6) Save outputs
# ---------------------------------------------------------------
pd.DataFrame([full_spectrum_summary]).to_csv(
    os.path.join(OUT_DIR, 'ANG_spectral_validation_full_spectrum.csv'), index=False)
df_region.to_csv(
    os.path.join(OUT_DIR, 'ANG_spectral_validation_per_region.csv'), index=False)
pd.DataFrame([pooled_distributional_summary]).to_csv(
    os.path.join(OUT_DIR, 'ANG_spectral_validation_pooled_distributional.csv'), index=False)

# Diagnostic plot: matched mean spectra + per-band absolute error
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
axes[0].plot(matched_wl, real_spec, color='blue', label='Real (mean)')
axes[0].plot(matched_wl, sim_spec, color='red', label='Simulated (mean)')
axes[0].set_ylabel('Reflectance')
axes[0].set_title('Wavelength-matched mean spectra (noisy/water bands excluded)')
axes[0].legend()
axes[0].grid(alpha=0.4)

axes[1].plot(matched_wl, np.abs(real_spec - sim_spec), color='black')
axes[1].set_xlabel('Wavelength (nm)')
axes[1].set_ylabel('|Real - Sim|')
axes[1].set_title(f"Absolute error per band  (RMSE={full_spectrum_summary['RMSE']}, "
                   f"SAM={full_spectrum_summary['SAM_deg']} deg)")
axes[1].grid(alpha=0.4)

plt.tight_layout()
fig_path = os.path.join(OUT_DIR, 'ANG_spectral_validation_diagnostic.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.show()

print(
    "\nSaved:\n"
    " - ANG_spectral_validation_full_spectrum.csv       (1-row table: SAM/RMSE/NRMSE/r, full matched spectrum)\n"
    " - ANG_spectral_validation_per_region.csv          (compact table: same metrics per VNIR/SWIR1/SWIR2)\n"
    " - ANG_spectral_validation_pooled_distributional.csv (RMSE/NRMSE/r using full pixel distributions)\n"
    " - ANG_spectral_validation_diagnostic.png          (matched mean spectra + per-band error plot)\n"
    f"in {OUT_DIR}"
)
