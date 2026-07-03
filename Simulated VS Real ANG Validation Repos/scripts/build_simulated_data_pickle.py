
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.io_utils import save_pickle
from src.envi_processing import (
    open_envi_image,
    open_envi_gt_image,
    wavelength,
    view_msi_image_color,
    get_reflectance_image,
    extract_rois,
    plot_pixel_spectrum,
    plot_roi_spectra,
    show_bbox_band,
)

import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


data_dir = Path(
    'D:/Drive F 5-11-2024/A_BiosCape_Project3/Bioscape Real data campaign/output_3mGSD'
)
basename = 'AVIRIS-NG'
save_path = Path(__file__).resolve().parent / 'simulateddata.pkl'

img_rad = open_envi_image(data_dir, basename)
view_msi_image_color(img_rad)
img_ref = get_reflectance_image(img_rad)
wl_sim = wavelength(data_dir, basename)

leaf_pixel_x = 345
leaf_pixel_y = 345
leaf_pixel_squeezed = plot_pixel_spectrum(img_ref, wl_sim, leaf_pixel_x, leaf_pixel_y)

coordinates = [
    (342, 342)
]
rois = []
width = 15
height = 15
line = 2
intensity = 2
image_bboxed = None

for coordinate in coordinates:
    (x, y) = coordinate
    (roi, image_bboxed) = extract_rois(img_ref, x, y, width, height, intensity, line)
    rois.append(roi)

sns.axes_style('whitegrid')
q = 3
show_bbox_band(image_bboxed, q=q)
plot_roi_spectra(rois, wl_sim, title='Spectral Footprint Mean in ROI Area')

basename_truth = 'AVIRIS-NG_truth'
truth_img, header_file = open_envi_gt_image(data_dir, basename_truth)
Labelled_data = truth_img.read_bands(0)

sns.axes_style('whitegrid')
fig = plt.figure(figsize=(12, 6))
q = 7
plt.imshow(Labelled_data, cmap='jet')
plt.colorbar()
plt.title('Ground Truth')

rois_gt = []
image_bboxed = None
for coordinate in coordinates:
    (x, y) = coordinate
    (roi_gt, image_bboxed) = extract_rois(truth_img, x, y, width, height, intensity, line)
    rois_gt.append(roi_gt)

Labelled_data_roi = roi_gt.read_bands(0)

sns.axes_style('whitegrid')
fig = plt.figure(figsize=(12, 6))
plt.imshow(Labelled_data_roi)
plt.colorbar()
plt.title(f'band - {q}')

wl_sim = [w * 1000 for w in wl_sim]
payload = {'roi': roi, 'wl_sim': wl_sim}
save_pickle(payload, save_path)
