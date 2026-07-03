import numpy as np
import matplotlib.pyplot as plt

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
import seaborn as sns

script_dir = os.path.dirname(os.path.abspath(__file__))
load_path = os.path.join(script_dir, 'data.pkl')

with open(load_path, 'rb') as f:
    data = pickle.load(f)

print(f"Loaded from: {load_path}")

sim_Green_2006 = data['sim_Green_2006']; sim_Red_2006 = data['sim_Red_2006']
sim_Rededge_2006 = data['sim_Rededge_2006']; sim_NIR_2006 = data['sim_NIR_2006']
real_Green_2006 = data['real_Green_2006']; real_Red_2006 = data['real_Red_2006']
real_Rededge_2006 = data['real_Rededge_2006']; real_NIR_2006 = data['real_NIR_2006']
sim_Green_2016 = data['sim_Green_2016']; sim_Red_2016 = data['sim_Red_2016']
sim_Rededge_2016 = data['sim_Rededge_2016']; sim_NIR_2016 = data['sim_NIR_2016']
real_Green_2016 = data['real_Green_2016']; real_Red_2016 = data['real_Red_2016']
real_Rededge_2016 = data['real_Rededge_2016']; real_NIR_2016 = data['real_NIR_2016']
sim_Green_2019 = data['sim_Green_2019']; sim_Red_2019 = data['sim_Red_2019']
sim_Rededge_2019 = data['sim_Rededge_2019']; sim_NIR_2019 = data['sim_NIR_2019']
real_Green_2019 = data['real_Green_2019']; real_Red_2019 = data['real_Red_2019']
real_Rededge_2019 = data['real_Rededge_2019']; real_NIR_2019 = data['real_NIR_2019']


def calculate_statistics(array):
    return {
        'mean': np.mean(array),
        'median': np.median(array),
        'std_dev': np.std(array),
        'min': np.min(array),
        'max': np.max(array)
    }


def calculate_percentage_difference(real, sim):
    return ((sim - real) / real) * 100 if real != 0 else None


def plot_band_comparison(real_2019, sim_2019, real_2016, sim_2016, real_2006, sim_2006, band_name):
    #save_path = r"D:\Drive F 5-11-2024\A_BiosCape_Project3\AVIRIS-NG simulated Images\Scene_Validation\Simulated VS Real Drone Validation Repos\output"
    save_path = os.path.join(REPO_ROOT, 'results', 'bands_3years')
    os.makedirs(save_path, exist_ok=True)
    datasets = {
        f"Real {band_name} 2006": calculate_statistics(real_2006),
        f"Sim {band_name} 2006": calculate_statistics(sim_2006),
        f"Real {band_name} 2016": calculate_statistics(real_2016),
        f"Sim {band_name} 2016": calculate_statistics(sim_2016),
        f"Real {band_name} 2019": calculate_statistics(real_2019),
        f"Sim {band_name} 2019": calculate_statistics(sim_2019)
    }
    colors = ['skyblue', 'olive'] * 4
    data_plot = [real_2006, sim_2006, real_2016, sim_2016, real_2019, sim_2019]
    labels = list(datasets.keys())
    plt.figure(figsize=(12, 8))
    boxplot = plt.boxplot(data_plot, labels=labels, patch_artist=True,
                          boxprops=dict(facecolor='lightblue', color='blue'),
                          medianprops=dict(color='red', linewidth=2.5),
                          whiskerprops=dict(color='blue', linewidth=2.5))
    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)
    plt.title(f"Comparison of Real vs Simulated {band_name} Reflectance\nof Different Post-Fire Areas", fontsize=18)
    plt.ylabel("Reflectance", fontsize=18)
    plt.xticks(rotation=15, fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    y_min, y_max = plt.ylim()
    plt.ylim(y_min, y_max * 1.2)
    for idx, label in enumerate(labels):
        stats = datasets[label]
        plt.text(idx + 1, y_max * 1.05, f"Mean: {stats['mean']:.4f}", horizontalalignment='center', fontsize=11, color='blue')
        plt.text(idx + 1, y_max * 0.98, f"Median: {stats['median']:.4f}", horizontalalignment='center', fontsize=11, color='black')
    plt.tight_layout()
    filepath = os.path.join(save_path, f"{band_name}_Reflectance_Boxplot.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {filepath}")


def stat(year, real_Green, sim_Green, real_Red, sim_Red, real_Rededge, sim_Rededge, real_NIR, sim_NIR):
    percentage_differences = {
        f"{label} vs Sim": calculate_percentage_difference(real.mean(), sim.mean())
        for label, real, sim in zip(
            ["Green", "Red", "Red Edge", "NIR"],
            [real_Green, real_Red, real_Rededge, real_NIR],
            [sim_Green, sim_Red, sim_Rededge, sim_NIR],
        )
    }
    return {"Year": year, "Percentage Differences": percentage_differences}


def plot_band_comparison_violin(real_2019, sim_2019, real_2016, sim_2016, real_2006, sim_2006, band_name):
    data = pd.DataFrame({
        "Reflectance": list(real_2006) + list(sim_2006) + list(real_2016) + list(sim_2016) + list(real_2019) + list(sim_2019),
        "Year": (["2006"] * len(real_2006) + ["2006"] * len(sim_2006) + ["2016"] * len(real_2016) + ["2016"] * len(sim_2016) + ["2019"] * len(real_2019) + ["2019"] * len(sim_2019)),
        "Type": (["Real"] * len(real_2006) + ["Sim"] * len(sim_2006) + ["Real"] * len(real_2016) + ["Sim"] * len(sim_2016) + ["Real"] * len(real_2019) + ["Sim"] * len(sim_2019))
    })
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=data, x="Year", y="Reflectance", hue="Type", split=True, inner="quartile", palette="Set2")
    plt.title(f"Real vs Simulated {band_name} Reflectance Distribution", fontsize=18)
    plt.ylabel("Reflectance", fontsize=16)
    plt.xlabel("Year", fontsize=16)
    plt.legend(title="", fontsize=14, facecolor='white', edgecolor='black')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


plot_band_comparison(real_Green_2019, sim_Green_2019, real_Green_2016, sim_Green_2016, real_Green_2006, sim_Green_2006, "Green")
plot_band_comparison(real_Red_2019, sim_Red_2019, real_Red_2016, sim_Red_2016, real_Red_2006, sim_Red_2006, "Red")
plot_band_comparison(real_Rededge_2019, sim_Rededge_2019, real_Rededge_2016, sim_Rededge_2016, real_Rededge_2006, sim_Rededge_2006, "Red Edge")
plot_band_comparison(real_NIR_2019, sim_NIR_2019, real_NIR_2016, sim_NIR_2016, real_NIR_2006, sim_NIR_2006, "NIR")

result_2019 = stat(2019, real_Green_2019, sim_Green_2019, real_Red_2019, sim_Red_2019, real_Rededge_2019, sim_Rededge_2019, real_NIR_2019, sim_NIR_2019)
result_2016 = stat(2016, real_Green_2016, sim_Green_2016, real_Red_2016, sim_Red_2016, real_Rededge_2016, sim_Rededge_2016, real_NIR_2016, sim_NIR_2016)
result_2006 = stat(2006, real_Green_2006, sim_Green_2006, real_Red_2006, sim_Red_2006, real_Rededge_2006, sim_Rededge_2006, real_NIR_2006, sim_NIR_2006)
print(result_2019["Percentage Differences"])
print(result_2016["Percentage Differences"])
print(result_2006["Percentage Differences"])

# plot_band_comparison_violin(real_Green_2019, sim_Green_2019, real_Green_2016, sim_Green_2016, real_Green_2006, sim_Green_2006, "Green")
# plot_band_comparison_violin(real_Red_2019, sim_Red_2019, real_Red_2016, sim_Red_2016, real_Red_2006, sim_Red_2006, "Red")
# plot_band_comparison_violin(real_Rededge_2019, sim_Rededge_2019, real_Rededge_2016, sim_Rededge_2016, real_Rededge_2006, sim_Rededge_2006, "Red Edge")
# plot_band_comparison_violin(real_NIR_2019, sim_NIR_2019, real_NIR_2016, sim_NIR_2016, real_NIR_2006, sim_NIR_2006, "NIR")
