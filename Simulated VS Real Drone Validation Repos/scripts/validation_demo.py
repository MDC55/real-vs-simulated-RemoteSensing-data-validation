from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
    
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from scipy.stats import ks_2samp
from pathlib import Path

from src.indices_and_metrics import compute_indices, mean_feature, std_feature, shannon_index, build_metric_dataframe
from src.validation_summary import run_validation

DATA_PKL_PATH = Path(__file__).resolve().parent / 'data.pkl'
with open(DATA_PKL_PATH, 'rb') as f:
    data = pickle.load(f)

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

save_path = os.path.join(REPO_ROOT, 'results','Validation_demo_res')
os.makedirs(save_path, exist_ok=True)

df_2006_real = build_metric_dataframe(real_Red_2006, real_Green_2006, real_NIR_2006, real_Rededge_2006, 2006)
df_2016_real = build_metric_dataframe(real_Red_2016, real_Green_2016, real_NIR_2016, real_Rededge_2016, 2016)
df_2019_real = build_metric_dataframe(real_Red_2019, real_Green_2019, real_NIR_2019, real_Rededge_2019, 2019)

df_2006_sim = build_metric_dataframe(sim_Red_2006, sim_Green_2006, sim_NIR_2006, sim_Rededge_2006, 2006)
df_2016_sim = build_metric_dataframe(sim_Red_2016, sim_Green_2016, sim_NIR_2016, sim_Rededge_2016, 2016)
df_2019_sim = build_metric_dataframe(sim_Red_2019, sim_Green_2019, sim_NIR_2019, sim_Rededge_2019, 2019)

df_real = [df_2006_real, df_2016_real, df_2019_real]
df_sim = [df_2006_sim, df_2016_sim, df_2019_sim]
years = [2006, 2016, 2019]
metrics = ['NDVI', 'CVI', 'GNDVI', 'NIRv', 'NormNIR', 'LCI', 'Red', 'Green', 'NIR', 'RE']

percent_cover_2006 = [12, 15, 25, 0.5, 13, 5, 4, 3, 0.5, 5, 1, 1]
percent_cover_2016 = [15, 23, 1, 6, 5, 10, 8, 1, 5, 2, 1, 2, 1]
percent_cover_2019 = [5, 5, 2, 25, 22, 5, 5, 5, 5, 3, 5, 3]
shannon_indices = [shannon_index(percent_cover_2006), shannon_index(percent_cover_2016), shannon_index(percent_cover_2019)]

for metric in metrics:
    for i, year in enumerate(years):
        ks, p = ks_2samp(df_real[i][metric].values, df_sim[i][metric].values)
        print(f"Year {year} | {metric:8s} KS={ks:.3f} (p={p:.3g})")

results_df = run_validation(df_real, df_sim, metrics, years)
results_df.to_csv(os.path.join(save_path, "validation_summary.csv"), index=False)
print(results_df.round(4))

for column_name in metrics:
    mean_feature_values_sim = mean_feature(df_sim, column_name)
    mean_feature_values_real = mean_feature(df_real, column_name)
    std_feature_values_sim = std_feature(df_sim, column_name)
    std_feature_values_real = std_feature(df_real, column_name)
    print(f"Mean values for Simulated {column_name}: {np.round(mean_feature_values_sim, 4)}")
    print(f"Mean values for Real {column_name}: {np.round(mean_feature_values_real, 4)}")
    print(f"Std values for Simulated {column_name}: {np.round(std_feature_values_sim, 4)}")
    print(f"Std values for Real {column_name}: {np.round(std_feature_values_real, 4)}")
    plt.figure(figsize=(8, 5))
    plt.plot(shannon_indices, mean_feature_values_sim, marker='o', linestyle='--', color='b', label='Simulated')
    plt.plot(shannon_indices, mean_feature_values_real, marker='o', linestyle='--', color='r', label='Real')
    plt.title(f"Shannon Diversity Index vs Mean {column_name} of Different Post-fire Areas", fontsize=16)
    plt.xlabel("Shannon Diversity Index", fontsize=14)
    plt.ylabel(f"Mean {column_name}", fontsize=14)
    plt.legend(fontsize=12, facecolor='white', edgecolor='black')
    plt.grid(True)
    plt.tight_layout()
    out_file = os.path.join(save_path, f"{column_name}_Shannon_vs_Mean.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved plot: {out_file}")
