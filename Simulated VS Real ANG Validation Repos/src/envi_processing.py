
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import rioxarray as rxr
from shapely.geometry import mapping
from spectral.io import envi


def open_envi_image(data_dir, basename):
    data_dir = Path(data_dir)
    hsi_hdr_file = data_dir / f"{basename}.img.hdr"
    hsi_img_file = data_dir / f"{basename}.img"
    img_envi = envi.open(str(hsi_hdr_file), str(hsi_img_file))
    img = img_envi.load()
    return img


def open_envi_gt_image(data_dir, basename):
    data_dir = Path(data_dir)
    hsi_hdr_file = data_dir / f"{basename}.img.hdr"
    hsi_img_file = data_dir / f"{basename}.img"
    img_envi = envi.open(str(hsi_hdr_file), str(hsi_img_file))
    img = img_envi.load()
    header_file = envi.read_envi_header(str(hsi_hdr_file))
    print(header_file)
    return img, header_file


def wavelength(data_dir, basename):
    data_dir = Path(data_dir)
    hsi_hdr_file = data_dir / f"{basename}.img.hdr"
    hsi_img_file = data_dir / f"{basename}.img"
    img_envi = envi.open(str(hsi_hdr_file), str(hsi_img_file))
    wl = img_envi.bands.centers
    return wl


def view_msi_image_color(hsi_img, show=True):
    hsi_img_b = hsi_img[:, :, 3]
    hsi_img_b_normalized = hsi_img_b / np.max(hsi_img_b)
    hsi_img_g = hsi_img[:, :, 2]
    hsi_img_g_normalized = hsi_img_g / np.max(hsi_img_g)
    hsi_img_r = hsi_img[:, :, 1]
    hsi_img_r_normalized = hsi_img_r / np.max(hsi_img_r)
    image = np.asarray(
        255 * np.concatenate(
            [hsi_img_b_normalized, hsi_img_g_normalized, hsi_img_r_normalized],
            axis=2,
        ),
        dtype=np.uint8,
    )
    if show:
        plt.imshow(image)
        plt.show()
    return image


def get_reflectance_image(
    input_radiance_img,
    calpanel_80=((359, 339), (359, 339)),
    calpanel_20=((340, 360), (339, 360)),
    refl_80=0.8,
    refl_20=0.2,
):
    (r1_calpanel_80, c1_calpanel_80), (r2_calpanel_80, c2_calpanel_80) = calpanel_80
    (r1_calpanel_20, c1_calpanel_20), (r2_calpanel_20, c2_calpanel_20) = calpanel_20

    avg_calpanel_80_radiance_vec = (
        input_radiance_img[r1_calpanel_80, c1_calpanel_80, :]
        + input_radiance_img[r2_calpanel_80, c2_calpanel_80, :]
    ) / 2

    avg_calpanel_20_radiance_vec = (
        input_radiance_img[r1_calpanel_20, c1_calpanel_20, :]
        + input_radiance_img[r2_calpanel_20, c2_calpanel_20, :]
    ) / 2

    a_vec = (avg_calpanel_80_radiance_vec - avg_calpanel_20_radiance_vec) / (refl_80 - refl_20)
    b_vec = (avg_calpanel_20_radiance_vec * refl_80 - avg_calpanel_80_radiance_vec * refl_20) / (refl_80 - refl_20)
    reflectance_img = (input_radiance_img - b_vec) / a_vec
    return reflectance_img


def extract_rois(arr, x, y, w, h, intensity, line):
    roi = arr[y:y + h, x:x + w, :]
    bounding_box = arr.copy()
    bounding_box[y - line:y, x - line:x + w + line, :] = intensity
    bounding_box[y:y + h, x - line:x, :] = intensity
    bounding_box[y + h:y + h + line, x - line:x + w + line, :] = intensity
    bounding_box[y:y + h, x + w:x + w + line, :] = intensity
    return roi, bounding_box


def plot_pixel_spectrum(img, wl, x, y):
    leaf_pixel = img[y:y + 1, x:x + 1, :]
    leaf_pixel_squeezed = np.squeeze(leaf_pixel)
    plt.plot(wl, leaf_pixel_squeezed)
    plt.title(f"Spectral Footprint\n(Pixel {x},{y})")
    plt.xlabel('Wavelength')
    plt.ylabel('Reflectance')
    plt.show()
    return leaf_pixel_squeezed


def plot_roi_spectra(rois, wl, title='Spectral Footprint\n Mean in ROI Area'):
    for i in range(len(rois)):
        roi = rois[i]
        intensity = []
        for b in range(roi.shape[2]):
            intensity.append(np.mean(roi[:, :, b]))
        plt.plot(wl, intensity, label='ROI {}'.format(i + 1))
    plt.legend(loc='upper left')
    plt.title(title)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.show()


def show_bbox_band(image_bboxed, q=3, title_prefix='band - '):
    plt.imshow(image_bboxed[:, :, q])
    plt.colorbar()
    plt.title(f'{title_prefix}{q}')
    plt.show()


def open_ang_memmap(hdr_path):
    hdr_path = Path(hdr_path)
    ds = envi.open(str(hdr_path))
    metadata = ds.metadata
    dat = ds.open_memmap(interleave='bip')
    wl = np.array([float(x) for x in ds.metadata['wavelength']])
    return ds, metadata, dat, wl


def print_ang_metadata(metadata):
    pixel_size = metadata.get('pixel size', 'Pixel size information not available')
    print('Pixel Size:', pixel_size)
    map_info = metadata.get('map info', 'Map info not available')
    print('Map Info:', map_info)


def make_ang_rgb(dat, bands=(60, 40, 30)):
    rgb = dat[..., np.array(bands)].copy()
    rgb[rgb == -9999] = np.nan
    rgb -= np.nanpercentile(rgb, 5, axis=(0, 1))[np.newaxis, np.newaxis, :]
    rgb /= np.nanpercentile(rgb, 95, axis=(0, 1))[np.newaxis, np.newaxis, :]
    return rgb


def plot_ang_sample_spectra(wl, dat, row=680, col=610, count=10, ylim=(0, 0.35)):
    plt.figure(figsize=(10, 5))
    for x in range(count):
        plt.plot(wl, dat[row, col - x, :])
    plt.plot(wl, dat[row, col, :], c='black')
    plt.ylim(list(ylim))
    plt.xlabel('Wavelength [nm]')
    plt.show()


def read_shapefile(shapefile_path):
    return gpd.read_file(Path(shapefile_path))


def clip_raster_and_extract_bands(raster_path, shapefiles):
    raster = rxr.open_rasterio(raster_path)
    geometries = shapefiles.geometry.apply(mapping)
    clipped_rasters = []
    for geometry in geometries:
        clipped = raster.rio.clip([geometry], shapefiles.crs)
        clipped_rasters.append(clipped)
    return clipped_rasters


def extract_first_clipped_cube(clipped_rasters):
    wl = clipped_rasters[0].wavelength.values
    data = clipped_rasters[0].data.T
    return data, wl


def plot_clipped_pixels(wl, data, index=0, ylim=(0, 0.35)):
    data_to_plot = data[index]
    for i in range(data_to_plot.shape[0]):
        plt.plot(wl, data_to_plot[i], label=f'pixel {i + 1}')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.title('Wavelength vs Reflectance Data')
    plt.legend()
    plt.grid()
    plt.ylim(list(ylim))
    plt.show()


def plot_mean_spectrum(wl, data, ylim=(0, 0.35), color='blue', label='Mean Reflectance'):
    mean_data = np.mean(data, axis=(0, 1))
    plt.figure(figsize=(10, 5))
    plt.plot(wl, mean_data, color=color, label=label)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Mean Reflectance')
    plt.title('Mean Reflectance vs. Wavelength')
    plt.legend()
    plt.grid()
    plt.ylim(list(ylim))
    plt.show()
    return mean_data
