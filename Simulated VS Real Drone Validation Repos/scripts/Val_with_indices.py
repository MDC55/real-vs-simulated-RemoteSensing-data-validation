from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#%%
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.indices_and_metrics import compute_indices, mean_feature, std_feature, shannon_index, build_metric_dataframe
from src.validation_stats import (
    ks_compare,
    mean_std_string,
    percent_diff,
    significance_label,
    bold_best,
    build_summary_table,
    build_journal_table,
)
from src.figures_indices import hist_intersection, plot_shannon_vs_mean

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

print(f"Loaded from: {DATA_PKL_PATH}")

# Percent cover values for different years
percent_cover_2006 = [12, 15, 25, 0.5, 13, 5, 4, 3, 0.5, 5, 1, 1]
percent_cover_2016 = [15, 23, 1, 6, 5, 10, 8, 1, 5, 2, 1, 2, 1]
percent_cover_2019 = [5, 5, 2, 25, 22, 5, 5, 5, 5, 3, 5, 3]

sh_2006 = shannon_index(percent_cover_2006)
sh_2016 = shannon_index(percent_cover_2016)
sh_2019 = shannon_index(percent_cover_2019)
shannon_indices = [sh_2006, sh_2016, sh_2019]

# Example with your 2006 real dataset
out = compute_indices(
    R=real_Red_2006,
    G=real_Green_2006,
    NIR=real_NIR_2006,
    RE=real_Rededge_2006,
)
print('NDVI:', out['NDVI'])
print('CVI:', out['CVI'])
print('EVI:', out['EVI'])

# Build dataframes preserving original logic
df_2006_sim = build_metric_dataframe(sim_Red_2006, sim_Green_2006, sim_NIR_2006, sim_Rededge_2006, 'Year 2006', compute_indices)
df_2016_sim = build_metric_dataframe(sim_Red_2016, sim_Green_2016, sim_NIR_2016, sim_Rededge_2016, 'Year 2016', compute_indices)
df_2019_sim = build_metric_dataframe(sim_Red_2019, sim_Green_2019, sim_NIR_2019, sim_Rededge_2019, 'Year 2019', compute_indices)
df_2006_real = build_metric_dataframe(real_Red_2006, real_Green_2006, real_NIR_2006, real_Rededge_2006, 'Year 2006', compute_indices)
df_2016_real = build_metric_dataframe(real_Red_2016, real_Green_2016, real_NIR_2016, real_Rededge_2016, 'Year 2016', compute_indices)
df_2019_real = build_metric_dataframe(real_Red_2019, real_Green_2019, real_NIR_2019, real_Rededge_2019, 'Year 2019', compute_indices)

df_sim = [df_2006_sim, df_2016_sim, df_2019_sim]
df_real = [df_2006_real, df_2016_real, df_2019_real]
years = ['Year 2006', 'Year 2016', 'Year 2019']
metrics = [col for col in df_2006_sim.columns if col != 'Year']

#%%  mean/std reporting and quick plots
plt.rcParams.update({'font.size': 16})
for column_name in metrics:
    mean_feature_values_sim = mean_feature(df_sim, column_name)
    print(f"Mean values for Simulated {column_name}: {np.round(mean_feature_values_sim, 4)}")
    mean_feature_values_real = mean_feature(df_real, column_name)
    print(f"Mean values for Real {column_name}: {np.round(mean_feature_values_real, 4)}")
    std_feature_values_sim = std_feature(df_sim, column_name)
    print(f"Std values for Simulated {column_name}: {np.round(std_feature_values_sim, 4)}")
    std_feature_values_real = std_feature(df_real, column_name)
    print(f"Std values for Real {column_name}: {np.round(std_feature_values_real, 4)}")

    plt.figure(figsize=(8, 5))
    plt.plot(shannon_indices, mean_feature_values_sim, marker='o', linestyle='--', color='b', label='Simulated')
    plt.plot(shannon_indices, mean_feature_values_real, marker='o', linestyle='--', color='r', label='Real')
    plt.title(f"Shannon Diversity Index vs Mean {column_name} of Different Post-fire Areas", fontsize=18)
    plt.xlabel('Shannon Diversity Index', fontsize=16)
    plt.ylabel(f'Mean {column_name}', fontsize=16)
    plt.legend(fontsize=14, facecolor='white', edgecolor='black')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()



#%% Summary tables
summary_save_path = Path(__file__).resolve().parents[1] / 'results' / 'tables'
summary_save_path.mkdir(parents=True, exist_ok=True)

summary_table = build_summary_table(df_real, df_sim, metrics, [2006, 2016, 2019], mean_feature, std_feature)
print(summary_table)
summary_table.to_csv(summary_save_path / 'summary_table.csv', index=False, encoding='utf-8-sig')

pivot_table = summary_table.pivot(index='Metric', columns='Year', values=['Real (mean ± std)', 'Sim (mean ± std)'])
print(pivot_table)
pivot_table.to_csv(summary_save_path / 'pivot_table.csv', encoding='utf-8-sig')

#%% Journal-style comparison table
journal_table = build_journal_table(df_real, df_sim, metrics, [2006, 2016, 2019], mean_feature, std_feature)
print(journal_table)
journal_table.to_csv(summary_save_path / 'journal_table.csv', index=False, encoding='utf-8-sig')
print(type(df_real))

#%% Histogram intersections:  multi-year 

hist_save_path = Path(__file__).resolve().parents[1] / 'results' / 'indices'
hist_save_path.mkdir(parents=True, exist_ok=True)
for metric in metrics:
    for i, y in enumerate([2006, 2016, 2019]):
        r = df_real[i][metric].values
        s = df_sim[i][metric].values
        filepath = hist_save_path / f'{metric}_{y}.png'
        
        intersection_val = hist_intersection(r, s, bins=64, title=f'{metric}', show=False)  # don't show yet
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.show()   # show after saving
        plt.close()
        
        print(f'{metric} {y} → Intersection = {intersection_val:.3f}')        
#save → show → close

#%% Shannon-vs-mean saved plots
shannon_save_path = Path(__file__).resolve().parents[1] / 'results' / 'indices_per_year'
shannon_save_path.mkdir(parents=True, exist_ok=True)
for column_name in metrics:
    mean_feature_values_sim = mean_feature(df_sim, column_name)
    mean_feature_values_real = mean_feature(df_real, column_name)
    std_feature_values_sim = std_feature(df_sim, column_name)
    std_feature_values_real = std_feature(df_real, column_name)
    print(f'\n--- {column_name} ---')
    print(f"Mean values for Simulated {column_name}: {np.round(mean_feature_values_sim, 4)}")
    print(f"Mean values for Real {column_name}: {np.round(mean_feature_values_real, 4)}")
    print(f"Std values for Simulated {column_name}: {np.round(std_feature_values_sim, 4)}")
    print(f"Std values for Real {column_name}: {np.round(std_feature_values_real, 4)}")
    filepath = shannon_save_path / f'{column_name}_Shannon_vs_Mean.png'
    plot_shannon_vs_mean(
        shannon_indices,
        mean_feature_values_sim,
        mean_feature_values_real,
        std_feature_values_sim,
        std_feature_values_real,
        years,
        column_name,
        save_path=filepath,
        show=True,
    )
    print(f'Saved plot: {filepath}')

#%% KS test block 

'''
KS statistic: measures the maximum difference between the cumulative distributions of 
real vs. simulated data.
Values close to 0 → distributions are very similar.
Values closer to 1 → distributions are very different.
p-value: probability that the two distributions are from the same population.
A small p (like 0.0) means we reject the null hypothesis of “same distribution.”

Key takeaways:
Best-matching metric distributions: NDVI, GNDVI, NormNIR, RE.
Moderate mismatches: CVI, NIRv, reflectance bands (Red, Green, NIR).
Worst mismatch: LCI — simulated values diverge strongly from real.
'''

ks_results = []

for metric in metrics:
    for i, y in enumerate(years):
        r = df_real[i][metric].values
        s = df_sim[i][metric].values
        ks, p = ks_compare(r, s)
        print(f"{y} | {metric:8s} KS={ks:.3f} (p={p:.3g})")
        ks_results.append({'Year': y, 'Metric': metric, 'KS': round(ks, 3), 'p_value': round(p, 6)})

ks_df = pd.DataFrame(ks_results)
ks_df.to_csv(summary_save_path / 'ks_results.csv', index=False, encoding='utf-8-sig')
print("KS results saved.")