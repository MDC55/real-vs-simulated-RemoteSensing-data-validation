from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
    

RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)    

from src.io_utils import load_pickle
from src.hyperspectral_indices import compute_hsi_indices
from src.validation_utils import (
    mask_noisy_wavelengths,
    build_summary_table,
    ks_compare,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


script_dir = Path(__file__).resolve().parent

simulated_data = load_pickle(script_dir / "simulateddata.pkl")
roi = simulated_data["roi"]
wl_sim = simulated_data["wl_sim"]

real_data = load_pickle(script_dir / "realdata.pkl")
data = real_data["data"]
wl = real_data["wl"]

# %% without filter
mean_data_real = np.mean(data, axis=(0, 1))
mean_data_sim = np.mean(roi, axis=(0, 1))

plt.figure(figsize=(10, 5))
plt.plot(wl, mean_data_real, color="blue", label="Mean Reflectance Real ANG")
plt.plot(wl_sim, mean_data_sim, color="red", label="Mean Reflectance Sim-ANG")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Mean Reflectance")
plt.title("Mean Reflectance vs. Wavelength")
plt.legend()
plt.grid()
plt.ylim([0, 0.35])
plt.show()

# %% without filter with added std confidence interval
mean_data_real = np.mean(data, axis=(0, 1))
std_data_real = np.std(data, axis=(0, 1))

mean_data_sim = np.mean(roi, axis=(0, 1))
std_data_sim = np.std(roi, axis=(0, 1))

plt.figure(figsize=(10, 5))
plt.plot(wl, mean_data_real, color="blue", label="Mean Reflectance Real ANG")
plt.fill_between(
    wl,
    mean_data_real - std_data_real,
    mean_data_real + std_data_real,
    color="blue",
    alpha=0.2,
    label="Std Dev Real",
)

plt.plot(wl_sim, mean_data_sim, color="red", label="Mean Reflectance Sim-ANG")
plt.fill_between(
    wl_sim,
    mean_data_sim - std_data_sim,
    mean_data_sim + std_data_sim,
    color="red",
    alpha=0.2,
    label="Std Dev Sim",
)

plt.xlabel("Wavelength (nm)")
plt.ylabel("Mean Reflectance")
plt.title("Mean Reflectance with Standard Deviation vs. Wavelength")
plt.legend()
plt.grid()
plt.ylim([-0.03, 0.50])
plt.show()

# %% wavelength filtering
R, S = mask_noisy_wavelengths(
    data_real=data,
    wavelength_real=wl,
    data_sim=roi,
    wavelength_sim=wl_sim,
)

wavelength_real = np.array(wl)
wavelength_sim = np.array(wl_sim)

mean_data_real = np.nanmean(R, axis=(0, 1))
mean_data_sim = np.nanmean(S, axis=(0, 1))

plt.figure(figsize=(10, 6))
plt.plot(wavelength_real, mean_data_real, label="Real Data", color="blue")
plt.plot(wavelength_sim, mean_data_sim, label="Simulated Data", color="red")
plt.xlabel("Wavelength")
plt.ylabel("Mean Intensity")
plt.title("Mean Intensity vs. Wavelength")
plt.legend()
plt.grid(True)
plt.show()

# %%
wl_R = wl
wl_S = wl_sim

indices_R = compute_hsi_indices(R, wl_R, sensor="real")
indices_S = compute_hsi_indices(S, wl_S, sensor="sim")

table = build_summary_table(indices_R, indices_S)
print(table)
table.to_csv(RESULTS_DIR / "real_vs_sim_summary_table.csv", index=False, encoding='utf-8-sig')

# %%
def hist_intersection(a, b, bins=64, title="Histogram Intersection"):
    ha, bins_edges = np.histogram(a, bins=bins, density=True)
    hb, _ = np.histogram(b, bins=bins_edges, density=True)
    intersection = np.sum(np.minimum(ha, hb)) * (bins_edges[1] - bins_edges[0])

    plt.figure(figsize=(6, 4))
    plt.hist(a, bins=bins_edges, alpha=0.5, label="Real", density=True, color="blue")
    plt.hist(b, bins=bins_edges, alpha=0.5, label="Sim", density=True, color="red")
    plt.title(f"{title}\nIntersection={intersection:.3f}")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

    return intersection


hist_results = {}

for m in indices_R.keys():
    real_vals = indices_R[m].flatten()
    sim_vals = indices_S[m].flatten()

    real_vals = real_vals[~np.isnan(real_vals)]
    sim_vals = sim_vals[~np.isnan(sim_vals)]

    intersection = hist_intersection(
        real_vals,
        sim_vals,
        bins=64,
        title=f"{m} Histogram Intersection",
    )
    hist_results[m] = intersection

hist_df = pd.DataFrame({
    "Metric": list(hist_results.keys()),
    "Histogram Intersection": list(hist_results.values())
})

print(hist_df)

table["Histogram Intersection"] = [
    hist_results[m] for m in table["Metric"]
]

# %%
'''
KS Statistic
0 → identical distributions
larger → more different

p-value
p < 0.05 → distributions significantly different
p ≥ 0.05 → cannot reject similarity

'''
ks_results = {}

for m in indices_R.keys():
    real_vals = indices_R[m].flatten()
    sim_vals = indices_S[m].flatten()

    real_vals = real_vals[~np.isnan(real_vals)]
    sim_vals = sim_vals[~np.isnan(sim_vals)]

    stat, pval = ks_compare(real_vals, sim_vals)

    ks_results[m] = {
        "KS Statistic": stat,
        "KS p-value": pval
    }

table["KS Statistic"] = [ks_results[m]["KS Statistic"] for m in table["Metric"]]
table["KS p-value"] = [ks_results[m]["KS p-value"] for m in table["Metric"]]

table["KS Statistic"] = table["KS Statistic"].round(4)
table["KS p-value"] = table["KS p-value"].apply(lambda x: f"{x:.2e}")

print(table)

# %% Barplot with curly braces and % difference
metric_names = list(indices_R.keys())

real_means = [np.nanmean(indices_R[m]) for m in metric_names]
sim_means = [np.nanmean(indices_S[m]) for m in metric_names]

real_std = [np.nanstd(indices_R[m]) for m in metric_names]
sim_std = [np.nanstd(indices_S[m]) for m in metric_names]

pct_diff = table["% Difference"].values

x = np.arange(len(metric_names))
width = 0.35
plt.rcParams.update({"font.size": 16})

fig, ax = plt.subplots(figsize=(12, 5))

ax.bar(
    x - width / 2,
    real_means,
    width,
    yerr=real_std,
    capsize=4,
    label="Real"
)

ax.bar(
    x + width / 2,
    sim_means,
    width,
    yerr=sim_std,
    capsize=4,
    label="Sim"
)


def add_brace(ax, x1, x2, y, text, brace_height=0.02):
    ax.plot(
        [x1, x1, x2, x2],
        [y, y + brace_height, y + brace_height, y],
        lw=1.5,
        c="black"
    )
    ax.text(
        (x1 + x2) / 2,
        y + brace_height * 1.4,
        text,
        ha="center",
        va="bottom",
        fontsize=12
    )


real_top = np.array(real_means) + np.array(real_std)
sim_top = np.array(sim_means) + np.array(sim_std)

y_max = np.max(np.maximum(real_top, sim_top))
offset = y_max * 0.08
ax.set_ylim(0, y_max * 1.25)

for i in range(len(metric_names)):
    left = x[i] - width / 2
    right = x[i] + width / 2

    top_real = real_means[i] + real_std[i]
    top_sim = sim_means[i] + sim_std[i]
    y = max(top_real, top_sim) + offset

    label = f"{pct_diff[i]:.1f}%"
    add_brace(ax, left, right, y, label)

ax.set_xticks(x)
ax.set_xticklabels(metric_names, rotation=45, ha="right")
ax.set_ylabel("Mean ± Std")
ax.set_title("Real vs Simulated Vegetation Indices", fontsize=16)
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "boxplot.png", dpi=300, bbox_inches="tight")
plt.show()


# %% some rnd with different pixels --> use filtered data
coordinates = [
    (3, 5),
    (6, 0),
    (2, 1),
    (7, 1),
    (4, 3),
    (3, 4),
    (5, 5),
    (0, 5),
    (4, 5),
    (0, 6),
    (7, 6),
]

spectral_values = np.array([S[coord[0], coord[1], :] for coord in coordinates])

mean_data_sim = np.nanmean(spectral_values, axis=0)
std_data_sim = np.nanstd(spectral_values, axis=0)

plt.figure(figsize=(10, 5))
plt.plot(wl_R, R[5, 10, :], color="blue", label="Mean Reflectance Real ANG")
plt.plot(wl_S, mean_data_sim, color="red", label="Mean Reflectance Sim-ANG")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Mean Reflectance")
plt.title("Mean Reflectance vs. Wavelength")
plt.legend()
plt.grid()
plt.ylim([0, 0.35])
plt.tight_layout()
plt.savefig(RESULTS_DIR / "random_pixels_filtered_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# %% final graph added on AGU poster --> use filtered data
mean_data_real = np.nanmean(R, axis=(0, 1))
std_data_real = np.nanstd(R, axis=(0, 1))

mean_data_real = R[5, 10, :]

mean_data_sim = np.nanmean(spectral_values, axis=0)
std_data_sim = np.nanstd(spectral_values, axis=0)

plt.rcParams.update({"font.size": 16})

plt.figure(figsize=(10, 5))

plt.plot(wl_R, mean_data_real, color="blue", label="Mean Reflectance Real ANG")
plt.fill_between(
    wl_R,
    mean_data_real - std_data_real,
    mean_data_real + std_data_real,
    color="blue",
    alpha=0.2,
    label="Std Dev Real",
)

plt.plot(wl_S, mean_data_sim, color="red", label="Mean Reflectance Sim-ANG")
plt.fill_between(
    wl_S,
    mean_data_sim - std_data_sim,
    mean_data_sim + std_data_sim,
    color="red",
    alpha=0.2,
    label="Std Dev Sim",
)

plt.xlabel("Wavelength (nm)")
plt.ylabel("Mean Reflectance")
plt.title("Simulated vs Real AVIRIS-NG Reflectance Comparision", fontsize=16)
plt.legend()
plt.grid()
plt.ylim([-0.02, 0.46])
plt.tight_layout()
plt.savefig(RESULTS_DIR / "agu_poster_filtered_comparison.png", dpi=300, bbox_inches="tight")
plt.show()