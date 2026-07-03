import os
from pathlib import Path
import pickle
import numpy as np
import spectral.io.envi as envi
import matplotlib.pyplot as plt
import geopandas as gpd
import rioxarray as rxr
from shapely.geometry import mapping


def open_envi_image(data_dir, basename):
    hsi_hdr_file = data_dir + '/' + f"{basename}.img.hdr"
    hsi_img_file = data_dir + '/' + f"{basename}.img"
    img_envi = envi.open(hsi_hdr_file, hsi_img_file)
    return img_envi.load()


def open_envi_gt_image(data_dir, basename):
    hsi_hdr_file = data_dir + '/' + f"{basename}.img.hdr"
    hsi_img_file = data_dir + '/' + f"{basename}.img"
    img_envi = envi.open(hsi_hdr_file, hsi_img_file)
    img = img_envi.load()
    header_file = envi.read_envi_header(hsi_hdr_file)
    return img, header_file


def wavelength(data_dir, basename):
    hsi_hdr_file = data_dir + '/' + f"{basename}.img.hdr"
    hsi_img_file = data_dir + '/' + f"{basename}.img"
    img_envi = envi.open(hsi_hdr_file, hsi_img_file)
    return img_envi.bands.centers


def view_msi_image_color(hsi_img):
    hsi_img_b = hsi_img[:, :, 3]
    hsi_img_b_normalized = hsi_img_b / np.max(hsi_img_b)
    hsi_img_g = hsi_img[:, :, 2]
    hsi_img_g_normalized = hsi_img_g / np.max(hsi_img_g)
    hsi_img_r = hsi_img[:, :, 1]
    hsi_img_r_normalized = hsi_img_r / np.max(hsi_img_r)
    image = np.asarray(255 * np.concatenate([hsi_img_b_normalized, hsi_img_g_normalized, hsi_img_r_normalized], axis=2), dtype=np.uint8)
    plt.imshow(image)
    plt.show()
    return image


def get_reflectance_image(input_radiance_img):
    r1_calpanel_80, c1_calpanel_80 = 456, 65
    r2_calpanel_80, c2_calpanel_80 = 450, 65
    r1_calpanel_20, c1_calpanel_20 = 20, 575
    r2_calpanel_20, c2_calpanel_20 = 25, 590
    avg_calpanel_80_radiance_vec = (input_radiance_img[r1_calpanel_80, c1_calpanel_80, :] + input_radiance_img[r2_calpanel_80, c2_calpanel_80, :]) / 2
    avg_calpanel_20_radiance_vec = (input_radiance_img[r1_calpanel_20, c1_calpanel_20, :] + input_radiance_img[r2_calpanel_20, c2_calpanel_20, :]) / 2
    a_vec = (avg_calpanel_80_radiance_vec - avg_calpanel_20_radiance_vec) / (0.8 - 0.2)
    b_vec = (avg_calpanel_20_radiance_vec * 0.8 - avg_calpanel_80_radiance_vec * 0.2) / (0.8 - 0.2)
    return (input_radiance_img - b_vec) / a_vec


def process_radiance_image(data_dir, basename='Drone', preview=True):
    input_radiance_img = open_envi_image(data_dir, basename)
    if preview:
        view_msi_image_color(input_radiance_img)
    img_ref = get_reflectance_image(input_radiance_img)
    wl = wavelength(data_dir, basename)
    return img_ref, wl


def plot_spectral_footprint(img_ref, wl, pixel_x, pixel_y):
    leaf_pixel = img_ref[pixel_y:pixel_y + 1, pixel_x:pixel_x + 1, :]
    leaf_pixel_squeezed = np.squeeze(leaf_pixel)
    plt.plot(wl, leaf_pixel_squeezed)
    plt.title(f'Spectral Footprint\n(Pixel {pixel_x},{pixel_y})')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.show()


def ground_truth_image(data_dir, basename='Drone_truth'):
    truth_img, header_file = open_envi_gt_image(data_dir, basename)
    labelled_data = truth_img.read_bands(0)
    plt.figure(figsize=(12, 6))
    plt.grid(False)
    plt.imshow(labelled_data, cmap='jet')
    plt.colorbar()
    plt.title('Ground Truth Labeled Image')
    plt.show()
    return truth_img, header_file


def extract_rois(arr, x, y, w, h, intensity=2, line=2):
    roi = arr[y:y + h, x:x + w, :]
    bounding_box = arr.copy()
    bounding_box[y - line:y, x - line:x + w + line, :] = intensity
    bounding_box[y:y + h, x - line:x, :] = intensity
    bounding_box[y + h:y + h + line, x - line:x + w + line, :] = intensity
    bounding_box[y:y + h, x + w:x + w + line, :] = intensity
    return roi, bounding_box


def GT_roi(data_dir, coordinates, width, height, intensity=2, line_width=2, basename='Drone_truth'):
    truth_img, header_file = open_envi_gt_image(data_dir, basename)
    rois_gt = []
    for coordinate in coordinates:
        x, y = coordinate
        roi_gt, image_bboxed = extract_rois(truth_img, x, y, width, height, intensity, line_width)
        rois_gt.append(roi_gt)
    labelled_data_roi = roi_gt.read_bands(0)
    plt.figure(figsize=(12, 6))
    plt.grid(False)
    plt.imshow(labelled_data_roi, cmap='jet')
    plt.colorbar()
    plt.title('Extracted ROI from Ground Truth Image')
    return roi_gt, header_file


def reflectance_img_roi(data_dir, coordinates, width, height, intensity=2, line_width=2):
    img_ref, wl = process_radiance_image(data_dir)
    rois_reflectance = []
    for coordinate in coordinates:
        x, y = coordinate
        roi, image_bboxed = extract_rois(img_ref, x, y, width, height, intensity, line_width)
        rois_reflectance.append(roi)
    plt.figure(figsize=(12, 6))
    plt.grid(False)
    q = 3
    plt.imshow(image_bboxed[:, :, q])
    plt.colorbar()
    plt.title(f'band - {q}')
    for i, roi in enumerate(rois_reflectance):
        intensity_vals = [np.mean(roi[:, :, b]) for b in range(roi.shape[2])]
        plt.plot(wl, intensity_vals, label=f'ROI {i + 1}')
    plt.legend(loc='upper left')
    plt.title('Spectral Footprint\n Mean in ROI Area')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.grid(False)
    plt.show()
    return roi


def clip_raster_and_extract_bands(raster_path, raster_filename, shapefiles):
    raster = rxr.open_rasterio(Path(raster_path, raster_filename), masked=True).squeeze()
    geometries = shapefiles.geometry.apply(mapping)
    clipped_rasters = []
    for geometry in geometries:
        clipped = raster.rio.clip([geometry], shapefiles.crs)
        clipped_rasters.append(clipped)
    band_data = []
    for band_index in range(4):
        band_values = [clipped_raster[band_index].values for clipped_raster in clipped_rasters]
        band_values = np.concatenate(band_values, axis=0)
        band_data.append(band_values)
    return np.stack(band_data, axis=-1)


def load_shapefile(path_str):
    return gpd.read_file(Path(path_str))


def remove_outliers_iqr(data):
    q1 = np.percentile(data, 25, axis=(0, 1), keepdims=True)
    q3 = np.percentile(data, 75, axis=(0, 1), keepdims=True)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return np.where((data < lower_bound) | (data > upper_bound), np.nan, data)


def nan_mask(Green, Red, Rededge, NIR):
    all_bands = np.stack([Green, Red, Rededge, NIR], axis=-1)
    mask = np.any(np.isnan(all_bands), axis=-1)
    return Green[~mask], Red[~mask], Rededge[~mask], NIR[~mask]


def save_data_pkl(data_dict, script_file, filename='data.pkl'):
    script_dir = os.path.dirname(os.path.abspath(script_file))
    save_path = os.path.join(script_dir, filename)
    with open(save_path, 'wb') as f:
        pickle.dump(data_dict, f)
    return save_path
