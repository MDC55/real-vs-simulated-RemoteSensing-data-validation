from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def hist_intersection(a, b, bins=64, title='Histogram Intersection', show=True):
    ha, bins_edges = np.histogram(a, bins=bins, density=True)
    hb, _ = np.histogram(b, bins=bins_edges, density=True)
    intersection = np.sum(np.minimum(ha, hb)) * (bins_edges[1] - bins_edges[0])
    plt.figure(figsize=(6, 4))
    plt.hist(a, bins=bins_edges, alpha=0.5, label='Real', density=True, color='blue')
    plt.hist(b, bins=bins_edges, alpha=0.5, label='Sim', density=True, color='red')
    plt.title(f"{title}\nIntersection={intersection:.3f}")
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    if show:
        plt.show()
    return intersection




def plot_shannon_vs_mean(shannon_indices, mean_feature_values_sim, mean_feature_values_real, std_feature_values_sim, std_feature_values_real, years, column_name, save_path=None, show=True):
    plt.figure(figsize=(10, 6))
    plt.plot(shannon_indices, mean_feature_values_sim, marker='o', markersize=8, linestyle='--', linewidth=2.5, color='r', label=f'Simulated Mean {column_name}')
    plt.fill_between(shannon_indices,
                     [m - s for m, s in zip(mean_feature_values_sim, std_feature_values_sim)],
                     [m + s for m, s in zip(mean_feature_values_sim, std_feature_values_sim)],
                     color='red', alpha=0.2, label='Simulated Std Dev')
    plt.plot(shannon_indices, mean_feature_values_real, marker='o', markersize=8, linestyle='--', linewidth=2.5, color='b', label=f'Real Mean {column_name}')
    plt.fill_between(shannon_indices,
                     [m - s for m, s in zip(mean_feature_values_real, std_feature_values_real)],
                     [m + s for m, s in zip(mean_feature_values_real, std_feature_values_real)],
                     color='blue', alpha=0.2, label='Real Std Dev')
    for i, year in enumerate(years):
        plt.text(shannon_indices[i] + 0.005, mean_feature_values_sim[i] + 0.005, f'{year}', fontsize=16, ha='right')
    plt.title(f'Shannon Diversity Index vs {column_name} of Different Post-fire Areas', fontsize=20)
    plt.xlabel('Shannon Diversity Index', fontsize=18)
    plt.ylabel(f'Mean {column_name}', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend()
    plt.style.use('default')
    plt.grid(True)
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
