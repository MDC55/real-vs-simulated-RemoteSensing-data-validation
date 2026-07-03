
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.io_utils import load_pickle
from src.hyperspectral_indices import compute_hsi_indices_uas
from src.validation_utils import (
    mask_noisy_wavelengths,
    build_summary_table,
)

import matplotlib.pyplot as plt
import numpy as np


script_dir = Path(__file__).resolve().parent

simulated_data = load_pickle(script_dir / 'simulateddata.pkl')
roi = simulated_data['roi']
wl_sim = simulated_data['wl_sim']

real_data = load_pickle(script_dir / 'realdata.pkl')
data = real_data['data']
wl = real_data['wl']


data_real, data_sim = mask_noisy_wavelengths(
    data_real=data,
    wavelength_real=wl,
    data_sim=roi,
    wavelength_sim=wl_sim,
)

wavelength_real = np.array(wl)
wavelength_sim = np.array(wl_sim)

mean_data_real = np.nanmean(data_real, axis=(0, 1))
mean_data_sim = np.nanmean(data_sim, axis=(0, 1))

plt.figure(figsize=(10, 6))
plt.plot(wavelength_real, mean_data_real, label='Real Data', color='blue')
plt.plot(wavelength_sim, mean_data_sim, label='Simulated Data', color='red')
plt.xlabel('Wavelength')
plt.ylabel('Mean Intensity')
plt.title('Mean Intensity vs. Wavelength')
plt.legend()
plt.grid(True)
plt.show()

indices_R = compute_hsi_indices_uas(data_real, wavelength_real)
indices_S = compute_hsi_indices_uas(data_sim, wavelength_sim)

table = build_summary_table(indices_R, indices_S)
print(table)
