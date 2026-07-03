import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, ttest_ind


def ks_compare(real, sim):
    stat, p = ks_2samp(real, sim)
    return stat, p


def mean_std_string(mean_arr, std_arr):
    return [f"{m:.4f} ± {s:.4f}" for m, s in zip(mean_arr, std_arr)]


def percent_diff(real_mean, sim_mean):
    if real_mean == 0:
        return np.nan
    return ((sim_mean - real_mean) / real_mean) * 100


def significance_label(p):
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'



def bold_best(real_mean, sim_mean, real_str, sim_str):
    """Bold the higher mean value."""
    if sim_mean > real_mean:
        return real_str, f"\\textbf{{{sim_str}}}"
    elif real_mean > sim_mean:
        return f"\\textbf{{{real_str}}}", sim_str
    else:
        return real_str, sim_str
    

def build_summary_table(df_real, df_sim, metrics, years, mean_feature, std_feature):
    rows = []
    for metric in metrics:
        mean_sim = mean_feature(df_sim, metric)
        std_sim = std_feature(df_sim, metric)
        mean_real = mean_feature(df_real, metric)
        std_real = std_feature(df_real, metric)
        for i, yr in enumerate(years):
            rows.append({
                'Year': yr,
                'Metric': metric,
                'Real (mean ± std)': f"{mean_real[i]:.4f} ± {std_real[i]:.4f}",
                'Sim (mean ± std)': f"{mean_sim[i]:.4f} ± {std_sim[i]:.4f}",
            })
    return pd.DataFrame(rows)


def build_journal_table(df_real, df_sim, metrics, years, mean_feature, std_feature):
    rows = []
    for metric in metrics:
        mean_sim = mean_feature(df_sim, metric)
        std_sim = std_feature(df_sim, metric)
        mean_real = mean_feature(df_real, metric)
        std_real = std_feature(df_real, metric)
        for i, yr in enumerate(years):
            real_vals = df_real[i][metric].values
            sim_vals = df_sim[i][metric].values
            if len(real_vals) > 1 and len(sim_vals) > 1:
                _, pval = ttest_ind(real_vals, sim_vals, equal_var=False)
            else:
                pval = np.nan
            sig = significance_label(pval)
            real_str = f"{mean_real[i]:.4f} ± {std_real[i]:.4f}"
            sim_str = f"{mean_sim[i]:.4f} ± {std_sim[i]:.4f}"
            real_fmt, sim_fmt = bold_best(mean_real[i], mean_sim[i], real_str, sim_str)
            pct = percent_diff(mean_real[i], mean_sim[i])
            rows.append({
                'Year': yr,
                'Metric': metric,
                'Real (mean±std)': real_fmt,
                'Sim (mean±std)': sim_fmt,
                '% Difference (Sim-Real)': f"{pct:.2f}",
                'p-value': f"{pval:.3e}",
                'Significance': sig,
            })
    return pd.DataFrame(rows)
