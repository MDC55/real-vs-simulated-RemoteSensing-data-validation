

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pathlib import Path
import pickle

DATA_PKL_PATH = Path(__file__).resolve().parent / 'data.pkl'
with open(DATA_PKL_PATH, 'rb') as f:
    data = pickle.load(f)

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle

sim_Green_2006 = data['sim_Green_2006']; sim_Red_2006 = data['sim_Red_2006']
sim_Rededge_2006 = data['sim_Rededge_2006']; sim_NIR_2006 = data['sim_NIR_2006']
real_Green_2006 = data['real_Green_2006']; real_Red_2006 = data['real_Red_2006']
real_Rededge_2006 = data['real_Rededge_2006']; real_NIR_2006 = data['real_NIR_2006']


def clean_array(arr):
    arr = np.asarray(arr).flatten()
    return arr[~np.isnan(arr)]


def calculate_statistics(array):
    array = clean_array(array)
    return {
        'mean': np.mean(array),
        'median': np.median(array),
        'std_dev': np.std(array),
        'min': np.min(array),
        'max': np.max(array),
        'n': len(array)
    }


def calculate_percentage_difference(real, sim):
    real_mean = np.mean(clean_array(real))
    sim_mean = np.mean(clean_array(sim))
    if real_mean == 0:
        return np.nan
    return abs((sim_mean - real_mean) / real_mean) * 100


def plot_2006_band_comparison(real_Green, sim_Green, real_Red, sim_Red, real_Rededge, sim_Rededge, real_NIR, sim_NIR):
   #save_path = r"D:\Drive F 5-11-2024\A_BiosCape_Project3\AVIRIS-NG simulated Images\Scene_Validation\Simulated VS Real Drone Validation Repos\output"
    save_path = os.path.join(REPO_ROOT, 'results', 'bands_1year')
    os.makedirs(save_path, exist_ok=True)
    datasets = [
        clean_array(real_Green), clean_array(sim_Green),
        clean_array(real_Red), clean_array(sim_Red),
        clean_array(real_Rededge), clean_array(sim_Rededge),
        clean_array(real_NIR), clean_array(sim_NIR),
    ]
    labels = [
        "Real Green 2006", "Sim Green 2006",
        "Real Red 2006", "Sim Red 2006",
        "Real Red Edge 2006", "Sim Red Edge 2006",
        "Real NIR 2006", "Sim NIR 2006",
    ]
    statistics = {label: calculate_statistics(data) for label, data in zip(labels, datasets)}
    pct_diff = {
        "Green": calculate_percentage_difference(real_Green, sim_Green),
        "Red": calculate_percentage_difference(real_Red, sim_Red),
        "Red Edge": calculate_percentage_difference(real_Rededge, sim_Rededge),
        "NIR": calculate_percentage_difference(real_NIR, sim_NIR),
    }
    plt.figure(figsize=(14, 8))
    colors = ['skyblue', 'olive'] * 4
    boxplot = plt.boxplot(datasets, labels=labels, patch_artist=True,
                          medianprops=dict(color='red', linewidth=2),
                          whiskerprops=dict(color='blue', linewidth=1.8))
    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)
    plt.title("2006 Reflectance Comparison: Real vs Simulated (All Bands)", fontsize=18)
    plt.ylabel("Reflectance", fontsize=16)
    plt.xticks(rotation=20, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    y_min, y_max = plt.ylim()
    plt.ylim(y_min, y_max * 1.25)
    for i, label in enumerate(labels):
        stats = statistics[label]
        plt.text(i + 1, y_max * 1.08, f"μ={stats['mean']:.4f}", ha='center', fontsize=10, color='blue')
    plt.tight_layout()
    filepath = os.path.join(save_path, "Reflectance_Boxplot_2006_AllBands.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {filepath}")
    print(pd.Series(pct_diff))
    return statistics, pct_diff


def plot_2006_band_subplots(real_Green, sim_Green, real_Red, sim_Red, real_Rededge, sim_Rededge, real_NIR, sim_NIR):
    #save_path = r"D:\Drive F 5-11-2024\A_BiosCape_Project3\AVIRIS-NG simulated Images\Scene_Validation\Simulated VS Real Drone Validation Repos\output"
    save_path = os.path.join(REPO_ROOT, 'results', 'bands_1year')
    os.makedirs(save_path, exist_ok=True)
    bands = [
        ("Green", real_Green, sim_Green),
        ("Red", real_Red, sim_Red),
        ("Red Edge", real_Rededge, sim_Rededge),
        ("NIR", real_NIR, sim_NIR),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 6))
    axes = axes.flatten()
    all_stats = {}
    pct_diff = {}
    for ax, (band_name, real, sim) in zip(axes, bands):
        real_c = clean_array(real)
        sim_c = clean_array(sim)
        data_plot = [real_c, sim_c]
        labels = ["Real", "Sim"]
        stats_real = calculate_statistics(real_c)
        stats_sim = calculate_statistics(sim_c)
        all_stats[band_name] = {"Real": stats_real, "Sim": stats_sim}
        pct_diff[band_name] = calculate_percentage_difference(real_c, sim_c)
        boxplot = ax.boxplot(data_plot, labels=labels, patch_artist=True,
                             medianprops=dict(color='red', linewidth=2),
                             whiskerprops=dict(color='blue', linewidth=1.5))
        colors = ['skyblue', 'olive']
        for patch, color in zip(boxplot['boxes'], colors):
            patch.set_facecolor(color)
        ax.set_title(f"{band_name}", fontsize=14)
        ax.set_ylabel("Reflectance")
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_max * 1.25)
        ax.text(1, y_max * 1.08, f"μ={stats_real['mean']:.4f}", ha='center', fontsize=10, color='blue')
        ax.text(2, y_max * 1.08, f"μ={stats_sim['mean']:.4f}", ha='center', fontsize=10, color='blue')
        ax.text(1.5, y_max * 1.18, f"%Δ={pct_diff[band_name]:.2f}%", ha='center', fontsize=11, color='darkgreen', fontweight='bold')
    fig.suptitle("Spectral Band Comparison: Real vs Simulated UAS data", fontsize=18, y=0.98)
    plt.tight_layout()
    filepath = os.path.join(save_path, "Reflectance_Subplots_2006_AllBands.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {filepath}")
    print(pd.Series(pct_diff))
    return all_stats, pct_diff


stats_2006, pct_2006 = plot_2006_band_comparison(
    real_Green_2006, sim_Green_2006,
    real_Red_2006, sim_Red_2006,
    real_Rededge_2006, sim_Rededge_2006,
    real_NIR_2006, sim_NIR_2006
)

stats_2006_sub, pct_2006_sub = plot_2006_band_subplots(
    real_Green_2006, sim_Green_2006,
    real_Red_2006, sim_Red_2006,
    real_Rededge_2006, sim_Rededge_2006,
    real_NIR_2006, sim_NIR_2006
)
