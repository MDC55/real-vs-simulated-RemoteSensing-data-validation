from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, pearsonr, spearmanr


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))


def mae(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))


def mbe(y_true, y_pred):
    return np.mean(np.array(y_pred) - np.array(y_true))


def concordance_correlation_coefficient(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    var_true, var_pred = np.var(y_true), np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    return (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2)


def ks_compare(real, sim):
    stat, p = ks_2samp(real, sim)
    return stat, p


def correlation_per_feature(df_sim_list, df_real_list, feature):
    correlations = []
    for sim_df, real_df in zip(df_sim_list, df_real_list):
        min_len = min(len(sim_df), len(real_df))
        sim_vals = sim_df[feature].iloc[:min_len]
        real_vals = real_df[feature].iloc[:min_len]
        corr = sim_vals.corr(real_vals)
        correlations.append(corr)
    return correlations


def validate_metric(real_df, sim_df, column_name, year):
    min_len = min(len(real_df), len(sim_df))
    y_true = real_df[column_name].iloc[:min_len]
    y_pred = sim_df[column_name].iloc[:min_len]
    ks_stat, ks_p = ks_compare(y_true, y_pred)
    return {
        "Year": year,
        "Metric": column_name,
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MBE": mbe(y_true, y_pred),
        "Pearson": pearsonr(y_true, y_pred)[0],
        "Spearman": spearmanr(y_true, y_pred)[0],
        "CCC": concordance_correlation_coefficient(y_true, y_pred),
        "KS_stat": ks_stat,
        "KS_pvalue": ks_p,
    }


def run_validation(df_real_list, df_sim_list, metrics, years):
    rows = []
    for real_df, sim_df, year in zip(df_real_list, df_sim_list, years):
        for metric in metrics:
            rows.append(validate_metric(real_df, sim_df, metric, year))
    return pd.DataFrame(rows)
