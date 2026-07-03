from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
    
"""
Upstream script that creates data.pkl for downstream validation scripts.
Hard-coded local paths are intentionally preserved.
"""
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data_preparation import (
    process_radiance_image,
    plot_spectral_footprint,
    ground_truth_image,
    GT_roi,
    reflectance_img_roi,
    clip_raster_and_extract_bands,
    load_shapefile,
    remove_outliers_iqr,
    nan_mask,
    save_data_pkl,
)

# Simulated ROI extraction
coordinates = [(120, 27)]
width = 405
height = 405

# 2019 simulated ROI
data_dir = 'D:/Drive F 5-11-2024/A_BiosCape_Project3/AVIRIS-NG simulated Images/Scene_Validation/2019/'
img_ref, wl = process_radiance_image(data_dir)
plot_spectral_footprint(img_ref, wl, 252, 172)
truth_img, header_file = ground_truth_image(data_dir)
roi_gt, header_file = GT_roi(data_dir, coordinates, width, height)
roi_Ref_2019 = reflectance_img_roi(data_dir, coordinates, width, height)

# 2016 simulated ROI
data_dir = 'D:/Drive F 5-11-2024/A_BiosCape_Project3/AVIRIS-NG simulated Images/Scene_Validation/2016/'
img_ref, wl = process_radiance_image(data_dir)
plot_spectral_footprint(img_ref, wl, 252, 172)
truth_img, header_file = ground_truth_image(data_dir)
roi_gt, header_file = GT_roi(data_dir, coordinates, width, height)
roi_Ref_2016 = reflectance_img_roi(data_dir, coordinates, width, height)

# 2006 simulated ROI
#data_dir='D:/Drive F 5-11-2024/A_BiosCape_Project3/AVIRIS-NG simulated Images/Scene_Validation/2006/'
data_dir='D:/Drive F 5-11-2024/A_BiosCape_Project3/AVIRIS-NG simulated Images/Scene_Validation/Simulated Scene_Validation_Ramesh/2006_new/'
img_ref, wl = process_radiance_image(data_dir)
plot_spectral_footprint(img_ref, wl, 252, 172)
truth_img, header_file = ground_truth_image(data_dir)
roi_gt, header_file = GT_roi(data_dir, coordinates, width, height)
roi_Ref_2006 = reflectance_img_roi(data_dir, coordinates, width, height)

# Real clipped rasters
# 2019
top_path = 'D:/Drive F 5-11-2024/2023 SA Fynbos Field Work/Drone data analysis code/'
raster_path = top_path + 'burn2019_lr/reflectance'
raster_filename = 'burn2019_lr_band_stack.tif'
shape_path = 'D:/Drive F 5-11-2024/A_BiosCape_Project3/AVIRIS-NG simulated Images/Scene_Validation/Shapefile_wholescene/2019/2019.shp'
shapefiles = load_shapefile(shape_path)
real_2019 = clip_raster_and_extract_bands(raster_path, raster_filename, shapefiles)

# 2016
top_path = 'D:/Drive F 5-11-2024/2023 SA Fynbos Field Work/Drone data analysis code/'
raster_path = top_path + 'burn2016_lr/reflectance'
raster_filename = 'burn2016_lr_band_stack.tif'
shape_path = 'D:/Drive F 5-11-2024/A_BiosCape_Project3/AVIRIS-NG simulated Images/Scene_Validation/Shapefile_wholescene/2016/2016.shp'
shapefiles = load_shapefile(shape_path)
real_2016 = clip_raster_and_extract_bands(raster_path, raster_filename, shapefiles)

# 2006
top_path = 'D:/Drive F 5-11-2024/2023 SA Fynbos Field Work/Drone data analysis code/'
raster_path = top_path + 'burnplot18_lr/reflectance'
raster_filename = 'burnplot18_lr_band_stack.tif'
shape_path = 'D:/Drive F 5-11-2024/A_BiosCape_Project3/AVIRIS-NG simulated Images/Scene_Validation/Shapefile_wholescene/2006/2006.shp'
shapefiles = load_shapefile(shape_path)
real_2006 = clip_raster_and_extract_bands(raster_path, raster_filename, shapefiles)

# Outlier removal
roi_Ref_2006_outliers_removed = remove_outliers_iqr(roi_Ref_2006)
roi_Ref_2019_outliers_removed = remove_outliers_iqr(roi_Ref_2019)
roi_Ref_2016_outliers_removed = remove_outliers_iqr(roi_Ref_2016)
real_2006_outliers_removed = remove_outliers_iqr(real_2006)
real_2019_outliers_removed = remove_outliers_iqr(real_2019)
real_2016_outliers_removed = remove_outliers_iqr(real_2016)

# Split bands
sim_Green_2006 = roi_Ref_2006_outliers_removed[:, :, 1]
sim_Red_2006 = roi_Ref_2006_outliers_removed[:, :, 2]
sim_Rededge_2006 = roi_Ref_2006_outliers_removed[:, :, 3]
sim_NIR_2006 = roi_Ref_2006_outliers_removed[:, :, 4]
sim_Green_2016 = roi_Ref_2016_outliers_removed[:, :, 1]
sim_Red_2016 = roi_Ref_2016_outliers_removed[:, :, 2]
sim_Rededge_2016 = roi_Ref_2016_outliers_removed[:, :, 3]
sim_NIR_2016 = roi_Ref_2016_outliers_removed[:, :, 4]
sim_Green_2019 = roi_Ref_2019_outliers_removed[:, :, 1]
sim_Red_2019 = roi_Ref_2019_outliers_removed[:, :, 2]
sim_Rededge_2019 = roi_Ref_2019_outliers_removed[:, :, 3]
sim_NIR_2019 = roi_Ref_2019_outliers_removed[:, :, 4]

real_Green_2006 = real_2006_outliers_removed[:, :, 0]
real_Red_2006 = real_2006_outliers_removed[:, :, 1]
real_Rededge_2006 = real_2006_outliers_removed[:, :, 2]
real_NIR_2006 = real_2006_outliers_removed[:, :, 3]
real_Green_2016 = real_2016_outliers_removed[:, :, 0]
real_Red_2016 = real_2016_outliers_removed[:, :, 1]
real_Rededge_2016 = real_2016_outliers_removed[:, :, 2]
real_NIR_2016 = real_2016_outliers_removed[:, :, 3]
real_Green_2019 = real_2019_outliers_removed[:, :, 0]
real_Red_2019 = real_2019_outliers_removed[:, :, 1]
real_Rededge_2019 = real_2019_outliers_removed[:, :, 2]
real_NIR_2019 = real_2019_outliers_removed[:, :, 3]

# Coordinated NaN masking
real_Green_2019, real_Red_2019, real_Rededge_2019, real_NIR_2019 = nan_mask(real_Green_2019, real_Red_2019, real_Rededge_2019, real_NIR_2019)
real_Green_2016, real_Red_2016, real_Rededge_2016, real_NIR_2016 = nan_mask(real_Green_2016, real_Red_2016, real_Rededge_2016, real_NIR_2016)
real_Green_2006, real_Red_2006, real_Rededge_2006, real_NIR_2006 = nan_mask(real_Green_2006, real_Red_2006, real_Rededge_2006, real_NIR_2006)
sim_Green_2019, sim_Red_2019, sim_Rededge_2019, sim_NIR_2019 = nan_mask(sim_Green_2019, sim_Red_2019, sim_Rededge_2019, sim_NIR_2019)
sim_Green_2016, sim_Red_2016, sim_Rededge_2016, sim_NIR_2016 = nan_mask(sim_Green_2016, sim_Red_2016, sim_Rededge_2016, sim_NIR_2016)
sim_Green_2006, sim_Red_2006, sim_Rededge_2006, sim_NIR_2006 = nan_mask(sim_Green_2006, sim_Red_2006, sim_Rededge_2006, sim_NIR_2006)

save_path = save_data_pkl({
    'sim_Green_2006': sim_Green_2006, 'sim_Red_2006': sim_Red_2006, 'sim_Rededge_2006': sim_Rededge_2006, 'sim_NIR_2006': sim_NIR_2006,
    'sim_Green_2016': sim_Green_2016, 'sim_Red_2016': sim_Red_2016, 'sim_Rededge_2016': sim_Rededge_2016, 'sim_NIR_2016': sim_NIR_2016,
    'sim_Green_2019': sim_Green_2019, 'sim_Red_2019': sim_Red_2019, 'sim_Rededge_2019': sim_Rededge_2019, 'sim_NIR_2019': sim_NIR_2019,
    'real_Green_2019': real_Green_2019, 'real_Red_2019': real_Red_2019, 'real_Rededge_2019': real_Rededge_2019, 'real_NIR_2019': real_NIR_2019,
    'real_Green_2016': real_Green_2016, 'real_Red_2016': real_Red_2016, 'real_Rededge_2016': real_Rededge_2016, 'real_NIR_2016': real_NIR_2016,
    'real_Green_2006': real_Green_2006, 'real_Red_2006': real_Red_2006, 'real_Rededge_2006': real_Rededge_2006, 'real_NIR_2006': real_NIR_2006,
}, __file__)

print(f'Saved data.pkl to: {save_path}')
