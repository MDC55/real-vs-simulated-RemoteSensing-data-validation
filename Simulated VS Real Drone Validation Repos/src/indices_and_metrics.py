from __future__ import annotations

import numpy as np
import pandas as pd


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


def mean_feature(dataframes, column_name):
    return [df[column_name].mean() for df in dataframes]


def std_feature(dataframes, column_name):
    return [df[column_name].std() for df in dataframes]


def shannon_index(abundance_values):
    total_abundance = sum(abundance_values)
    relative_abundance = [abundance / total_abundance for abundance in abundance_values]
    sh_idx = -sum(p * np.log(p) if p != 0 else 0 for p in relative_abundance)
    print(f"Shannon Diversity Index: {sh_idx}")
    return sh_idx



def compute_indices(R, G, NIR, RE):
    """
    Compute a wide range of vegetation indices.
    Parameters:
        R   = Red band
        G   = Green band
        NIR = Near Infrared band
        RE  = RedEdge band
    Returns:
        dict of vegetation indices
    """
    eps = 0#1e-6  # to avoid division by zero
    indices = {}

    # Core vegetation indices
    indices["NDVI"]   = (NIR - R) / (NIR + R + eps)
    indices["ReCI"]   = (NIR / (RE + eps)) - 1
    indices["NDRE"]   = (NIR - RE) / (NIR + RE + eps)
    indices["GNDVI"]  = (NIR - G) / (NIR + G + eps)
    indices["OSAVI"]  = (NIR - R) / (NIR + R + 0.16)
    indices["GCI"]    = (NIR / (G + eps)) - 1
    indices["SR"]     = NIR / (R + eps)
    indices["MSR"]    = ((NIR / (R + eps)) - 1) / np.sqrt((NIR / (R + eps)) + 1)
    indices["RDVI"]   = np.sqrt((NIR - R) / (NIR + R + eps))
    indices["NDWI"]   = (G - NIR) / (G + NIR + eps)

    # Simple differences/ratios
    indices["DVI"]    = NIR - R
    indices["GDVI"]   = NIR - G
    indices["RVI"]    = NIR / (G + eps)  # also called GRVI
    indices["WDRVI"]  = (0.05*NIR - R) / (0.05*NIR + R + eps)

    # Soil adjusted
    indices["GSAVI"]  = 1.5*((NIR - G) / (NIR + G + 0.5))
    indices["IPVI"]   = NIR / (NIR + R + eps)
    indices["MSAVI2"] = (2*NIR + 1 - np.sqrt((2*NIR + 1)**2 - 8*(NIR - R))) / 2
    indices["NLI"]    = (NIR**2 - R) / (NIR**2 + R + eps)
    indices["SAVI"]   = 1.5*(NIR - R) / (NIR + R + 0.5)
    indices["TDVI"]   = 1.5*((NIR - R) / (np.sqrt(NIR**2) + R + 0.5))
    indices["BAI"]    = 1 / (((0.1 - R)**2) + ((0.06 - NIR)**2) + eps)

    # Chlorophyll/anthocyanin related
    indices["AVI"]    = (NIR * (1 - R) * (NIR - R))**(1/3)
    indices["CVI"]    = (NIR * R) / (G**2 + eps)
    indices["DSWI4"]  = G / (R + eps)
    indices["GRNDVI"] = (NIR - (G+R)) / (NIR + (G+R) + eps)
    indices["MCARI1"] = 1.2*(2.5*(NIR - R) - 1.3*(NIR - G))
    indices["NGRDI"]  = (G - R) / (G + R + eps)
    indices["NIRv"]   = ((NIR - R) / (NIR + R + eps)) * NIR
    indices["NormG"]  = G / (NIR + G + R + eps)
    indices["NormNIR"]= NIR / (NIR + G + R + eps)
    indices["NormR"]  = R / (NIR + G + R + eps)
    indices["RGRI"]   = R / (G + eps)
    indices["RI"]     = (R - G) / (R + G + eps)
    indices["SR2"]    = NIR / (G + eps)
    indices["VIG"]    = (G - R) / (G + R + eps)
    indices["TVI"]    = np.sqrt(((NIR - R) / (NIR + R + eps)) + 0.5)
    indices["TriVI"]  = 0.5*(120*(NIR - G) - 200*(R - G))
    indices["MTVI1"]  = 1.2*(1.2*(NIR - G) - 2.5*(R - G))

    # Red-edge based
    indices["MSR_RE"]   = ((NIR / (RE + eps)) - 1) / np.sqrt((NIR / (RE + eps)) + 1)
    indices["MGRVI"]    = (G**2 - R**2) / (G**2 + R**2 + eps)
    indices["RERVI"]    = NIR / (RE + eps)
    indices["sCCCI"]    = indices["NDRE"] / (indices["NDVI"] + eps)
    indices["EVI2"]     = (NIR - R) / (1 + NIR + 2.4*R + eps)
    indices["MTCl"]     = (NIR - RE) / (RE - R + eps)
    indices["RESR"]     = NIR / (RE + eps)
    indices["PSRI"]     = (R - G) / (RE + eps)
    indices["M3Cl"]     = (NIR + R - RE) / (NIR - R + RE + eps)
    indices["SRrr"]     = RE / (R + eps)
    indices["REP"]      = R + (NIR / 2)
    indices["RENDVI"]   = (RE - R) / (RE + R + eps)
    indices["TCARI"]    = 3*((RE - R) - 0.2*(RE - G)*(RE / (R + eps)))
    indices["MCARI"]    = ((RE - R) - 0.2*(RE - G)) * (RE / (R + eps))
    indices["LCI"]      = (NIR - RE) / (NIR + R + eps)

    # Enhanced vegetation indices
    indices["EVI"]    = 2.5*(NIR - R) / (NIR + 6*R - 7.5*G + 1)
    indices["MEVI"]   = 2.5*(NIR - RE) / (NIR + 6*RE - 7.5*G + 1)
    indices["SARE"]   = (NIR - RE) / (NIR - RE + 0.25) + 0.25
    indices["RTVI"]   = 100*(NIR - RE) - 10*(NIR - G)
    indices["REGNDVI"]= (RE - G) / (RE + G + eps)
    indices["ARI2"]   = NIR * ((1/(G + eps)) - (1/(RE + eps)))
    indices["ratio1"] = indices["MCARI"] / (indices["OSAVI"] + eps)
    indices["ratio2"] = indices["TCARI"] / (indices["OSAVI"] + eps)
    indices["GDVI_sq"]= (NIR**2 - R**2) / (NIR**2 + R**2 + eps)

    return indices


def build_metric_dataframe(R, G, NIR, RE, year_label, compute_indices_func=compute_indices):
    indices = compute_indices_func(R, G, NIR, RE)
    df = pd.DataFrame({
        'Red': R,
        'Green': G,
        'NIR': NIR,
        'RE': RE,
        'Year': [year_label] * len(R)
    })
    for idx_name, idx_values in indices.items():
        df[idx_name] = idx_values
    return df