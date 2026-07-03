
import numpy as np


def band_at(wl, target):
    return int(np.argmin(np.abs(np.array(wl) - target)))


def get_band(cube, wl, target):
    return cube[:, :, band_at(wl, target)]


def roi_stats(arr):
    arr_flat = arr.flatten()
    mean = np.nanmean(arr_flat)
    std = np.nanstd(arr_flat)
    return f"{mean:.3f} ± {std:.3f}"


def compute_hsi_indices_uas(cube, wl):
    eps = 0
    indices = {}
    R = get_band(cube, wl, 670)
    G = get_band(cube, wl, 550)
    NIR = get_band(cube, wl, 800)
    RE = get_band(cube, wl, 720)
    indices['NDVI'] = (NIR - R) / (NIR + R + eps)
    indices['ReCI'] = (NIR / (RE + eps)) - 1
    indices['NDRE'] = (NIR - RE) / (NIR + RE + eps)
    indices['GNDVI'] = (NIR - G) / (NIR + G + eps)
    indices['OSAVI'] = (NIR - R) / (NIR + R + 0.16)
    indices['GCI'] = (NIR / (G + eps)) - 1
    indices['SR'] = NIR / (R + eps)
    indices['MSR'] = ((NIR / (R + eps)) - 1) / np.sqrt((NIR / (R + eps)) + 1)
    indices['RDVI'] = (NIR - R) / np.sqrt(NIR + R + eps)
    indices['NDWI'] = (G - NIR) / (G + NIR + eps)
    indices['DVI'] = NIR - R
    indices['GDVI'] = NIR - G
    indices['RVI'] = NIR / (G + eps)
    indices['WDRVI'] = (0.05 * NIR - R) / (0.05 * NIR + R + eps)
    indices['GSAVI'] = 1.5 * ((NIR - G) / (NIR + G + 0.5))
    indices['IPVI'] = NIR / (NIR + R + eps)
    indices['MSAVI2'] = (2 * NIR + 1 - np.sqrt((2 * NIR + 1) ** 2 - 8 * (NIR - R))) / 2
    indices['NLI'] = (NIR ** 2 - R) / (NIR ** 2 + R + eps)
    indices['SAVI'] = 1.5 * (NIR - R) / (NIR + R + 0.5)
    indices['TDVI'] = 1.5 * ((NIR - R) / (np.abs(NIR) + R + 0.5))
    indices['BAI'] = 1 / (((0.1 - R) ** 2) + ((0.06 - NIR) ** 2) + eps)
    indices['AVI'] = (NIR * (1 - R) * (NIR - R)) ** (1 / 3)
    indices['CVI'] = (NIR * R) / (G ** 2 + eps)
    indices['DSWI4'] = G / (R + eps)
    indices['GRNDVI'] = (NIR - (G + R)) / (NIR + (G + R) + eps)
    indices['MCARI1'] = 1.2 * (2.5 * (NIR - R) - 1.3 * (NIR - G))
    indices['NGRDI'] = (G - R) / (G + R + eps)
    indices['NIRv'] = ((NIR - R) / (NIR + R + eps)) * NIR
    indices['NormG'] = G / (NIR + G + R + eps)
    indices['NormNIR'] = NIR / (NIR + G + R + eps)
    indices['NormR'] = R / (NIR + G + R + eps)
    indices['RGRI'] = R / (G + eps)
    indices['RI'] = (R - G) / (R + G + eps)
    indices['SR2'] = NIR / (G + eps)
    indices['VIG'] = (G - R) / (G + R + eps)
    indices['TVI'] = np.sqrt(((NIR - R) / (NIR + R + eps)) + 0.5)
    indices['TriVI'] = 0.5 * (120 * (NIR - G) - 200 * (R - G))
    indices['MTVI1'] = 1.2 * (1.2 * (NIR - G) - 2.5 * (R - G))
    indices['MSR_RE'] = ((NIR / (RE + eps)) - 1) / np.sqrt((NIR / (RE + eps)) + 1)
    indices['MGRVI'] = (G ** 2 - R ** 2) / (G ** 2 + R ** 2 + eps)
    indices['RERVI'] = NIR / (RE + eps)
    indices['sCCCI'] = indices['NDRE'] / (indices['NDVI'] + eps)
    indices['EVI2'] = (NIR - R) / (1 + NIR + 2.4 * R + eps)
    indices['MTCl'] = (NIR - RE) / (RE - R + eps)
    indices['RESR'] = NIR / (RE + eps)
    indices['PSRI'] = (R - G) / (RE + eps)
    indices['M3Cl'] = (NIR + R - RE) / (NIR - R + RE + eps)
    indices['SRrr'] = RE / (R + eps)
    indices['RENDVI'] = (RE - R) / (RE + R + eps)
    indices['TCARI'] = 3 * ((RE - R) - 0.2 * (RE - G) * (RE / (R + eps)))
    indices['MCARI'] = ((RE - R) - 0.2 * (RE - G)) * (RE / (R + eps))
    indices['LCI'] = (NIR - RE) / (NIR + R + eps)
    indices['EVI'] = 2.5 * (NIR - R) / (NIR + 6 * R - 7.5 * G + 1)
    indices['MEVI'] = 2.5 * (NIR - RE) / (NIR + 6 * RE - 7.5 * G + 1)
    indices['SARE'] = (NIR - RE) / (NIR - RE + 0.25) + 0.25
    indices['RTVI'] = 100 * (NIR - RE) - 10 * (NIR - G)
    indices['REGNDVI'] = (RE - G) / (RE + G + eps)
    indices['ARI2'] = NIR * ((1 / (G + eps)) - (1 / (RE + eps)))
    indices['ratio1'] = indices['MCARI'] / (indices['OSAVI'] + eps)
    indices['ratio2'] = indices['TCARI'] / (indices['OSAVI'] + eps)
    indices['GDVI_sq'] = (NIR ** 2 - R ** 2) / (NIR ** 2 + R ** 2 + eps)
    return indices


def compute_hsi_indices(cube, wl, sensor='real'):
    if sensor == 'real':
        red_wl = 670
        nir_wl = 800
    else:
        red_wl = 700
        nir_wl = 880

    R = get_band(cube, wl, red_wl)
    G = get_band(cube, wl, 550)
    NIR = get_band(cube, wl, nir_wl)
    RE = get_band(cube, wl, 720)
    B = get_band(cube, wl, 480)
    eps = 0

    indices = {}
    indices['NDVI'] = (NIR - R) / (NIR + R)
    indices['NDRE'] = (NIR - RE) / (NIR + RE)
    indices['GNDVI'] = (NIR - G) / (NIR + G + eps)
    indices['NormNIR'] = NIR / (NIR + G + R)
    indices['LCI'] = (NIR - RE) / (NIR + R)
    indices['EVI'] = 2.5 * (NIR - R) / (NIR + 6 * R - 7.5 * B + 1)
    indices['SAVI'] = ((1 + 0.5) * (NIR - R)) / (NIR + R + 0.5)
    indices['ARVI'] = (NIR - (2 * R - B)) / (NIR + (2 * R - B))
    indices['RENDVI'] = (NIR - RE) / (NIR + RE)
    indices['ReCI'] = (NIR / (RE + eps)) - 1
    indices['TVI'] = np.sqrt(((NIR - R) / (NIR + R + eps)) + 0.5)
    indices['RESR'] = NIR / (RE + eps)
    indices['MEVI'] = 2.5 * (NIR - RE) / (NIR + 6 * RE - 7.5 * G + 1)
    indices['SARE'] = (NIR - RE) / (NIR - RE + 0.25) + 0.25
    indices['GDVI'] = (NIR ** 2 - R ** 2) / (NIR ** 2 + R ** 2 + eps)
    indices['IPVI'] = NIR / (NIR + R + eps)
    return indices

