
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt


def mask_noisy_wavelengths(data_real, wavelength_real, data_sim, wavelength_sim):
    data_real = data_real.copy()
    data_sim = data_sim.copy()

    wavelength_real = np.array(wavelength_real)
    wavelength_sim = np.array(wavelength_sim)

    data_real[:, :, (1350 <= wavelength_real) & (wavelength_real <= 1500)] = np.nan
    data_real[:, :, (1796 <= wavelength_real) & (wavelength_real <= 1980)] = np.nan
    data_real[:, :, (wavelength_real > 2480)] = np.nan

    data_sim[:, :, (1350 <= wavelength_sim) & (wavelength_sim <= 1500)] = np.nan
    data_sim[:, :, (1790 <= wavelength_sim) & (wavelength_sim <= 1980)] = np.nan
    data_sim[:, :, (wavelength_sim > 2480)] = np.nan

    return data_real, data_sim


def roi_stats(arr, decimals=4):
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def roi_mean(arr):
    return np.nanmean(arr)


def percent_diff(real_mean, sim_mean):
    if real_mean == 0 or np.isnan(real_mean):
        return np.nan
    return abs((sim_mean - real_mean) / real_mean) * 100


def build_summary_table(indices_R, indices_S):
    real_means = {m: roi_mean(indices_R[m]) for m in indices_R}
    sim_means = {m: roi_mean(indices_S[m]) for m in indices_S}

    table = pd.DataFrame({
        'Metric': list(indices_R.keys()),
        'Real (mean ± std)': [roi_stats(indices_R[m]) for m in indices_R],
        'Sim (mean ± std)': [roi_stats(indices_S[m]) for m in indices_S],
        'Δ (Sim − Real)': [sim_means[m] - real_means[m] for m in indices_R],
        '% Difference': [percent_diff(real_means[m], sim_means[m]) for m in indices_R],
    })
    table['Δ (Sim − Real)'] = table['Δ (Sim − Real)'].round(4)
    table['% Difference'] = table['% Difference'].round(2)
    return table


def hist_intersection(a, b, bins=64, title='Histogram Intersection', show=True):
    ha, bins_edges = np.histogram(a, bins=bins, density=True)
    hb, _ = np.histogram(b, bins=bins_edges, density=True)
    intersection = np.sum(np.minimum(ha, hb)) * (bins_edges[1] - bins_edges[0])

    if show:
        plt.figure(figsize=(6, 4))
        plt.hist(a, bins=bins_edges, alpha=0.5, label='Real', density=True, color='blue')
        plt.hist(b, bins=bins_edges, alpha=0.5, label='Sim', density=True, color='red')
        plt.title(f"{title}Intersection={intersection:.3f}")
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    return intersection


def compute_histogram_intersections(indices_R, indices_S, bins=64, show=False):
    hist_results = {}
    for m in indices_R.keys():
        real_vals = indices_R[m].flatten()
        sim_vals = indices_S[m].flatten()
        real_vals = real_vals[~np.isnan(real_vals)]
        sim_vals = sim_vals[~np.isnan(sim_vals)]
        hist_results[m] = hist_intersection(real_vals, sim_vals, bins=bins, title=f'{m} Histogram Intersection', show=show)
    return hist_results


def ks_compare(real, sim):
    stat, p = ks_2samp(real, sim)
    return stat, p


def compute_ks_results(indices_R, indices_S):
    ks_results = {}
    for m in indices_R.keys():
        real_vals = indices_R[m].flatten()
        sim_vals = indices_S[m].flatten()
        real_vals = real_vals[~np.isnan(real_vals)]
        sim_vals = sim_vals[~np.isnan(sim_vals)]
        stat, pval = ks_compare(real_vals, sim_vals)
        ks_results[m] = {'KS Statistic': stat, 'KS p-value': pval}
    return ks_results


def add_histogram_results_to_table(table, hist_results):
    table = table.copy()
    table['Histogram Intersection'] = [hist_results[m] for m in table['Metric']]
    return table


def add_ks_results_to_table(table, ks_results):
    table = table.copy()
    table['KS Statistic'] = [ks_results[m]['KS Statistic'] for m in table['Metric']]
    table['KS p-value'] = [ks_results[m]['KS p-value'] for m in table['Metric']]
    table['KS Statistic'] = table['KS Statistic'].round(4)
    table['KS p-value'] = table['KS p-value'].apply(lambda x: f'{x:.2e}')
    return table
