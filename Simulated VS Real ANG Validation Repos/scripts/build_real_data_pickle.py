
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.io_utils import save_pickle
from src.envi_processing import (
    open_ang_memmap,
    print_ang_metadata,
    make_ang_rgb,
    plot_ang_sample_spectra,
    read_shapefile,
    clip_raster_and_extract_bands,
    extract_first_clipped_cube,
    plot_clipped_pixels,
    plot_mean_spectrum,
)

import matplotlib.pyplot as plt
from pathlib import Path


data_dir = Path(
    'D:/Drive F 5-11-2024/A_BiosCape_Project3/Bioscape Real data campaign/ang20231117t123306_003_L2A_OE_main_27577724_RFL_ORT'
)
hdr_file = data_dir / 'ang20231117t123306_003_L2A_OE_main_27577724_RFL_ORT.hdr'
raster_path = data_dir / 'ang20231117t123306_003_L2A_OE_main_27577724_RFL_ORT'
shapefile_path = Path(
    'D:/Drive F 5-11-2024/A_BiosCape_Project3/Bioscape Real data campaign/2019/2019_burn_yr.shp'
)
save_path = Path(__file__).resolve().parent / 'realdata.pkl'

ds, metadata, dat, wl = open_ang_memmap(hdr_file)
print_ang_metadata(metadata)

rgb = make_ang_rgb(dat, bands=(60, 40, 30))
plt.figure(figsize=(15, 15))
plt.scatter([680], [610])
plt.scatter([670], [610])
plt.imshow(rgb)
plt.show()

plot_ang_sample_spectra(wl, dat, row=680, col=610, count=10, ylim=(0, 0.35))

shapefiles = read_shapefile(shapefile_path)
clipped_rasters = clip_raster_and_extract_bands(raster_path, shapefiles)
data, wl = extract_first_clipped_cube(clipped_rasters)

plot_clipped_pixels(wl, data, index=0, ylim=(0, 0.35))
plot_mean_spectrum(wl, data, ylim=(0, 0.35), color='blue', label='Mean Reflectance')

payload = {'data': data, 'wl': wl}
save_pickle(payload, save_path)
