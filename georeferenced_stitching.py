import os, sys, math, warnings
import cv2
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_gcps
from rasterio.control import GroundControlPoint
from tqdm import tqdm
from pyproj import Proj, Transformer
import micasense.metadata as metadata
from pathlib import Path

# Suppress NotGeoreferencedWarning when loading individual captures
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

def calculate_fov(focal_length_mm, pixels, pixel_size_mm=0.00375):
    sensor_size_mm = pixels * pixel_size_mm
    fov_rad = 2 * math.atan(sensor_size_mm / (2 * focal_length_mm))
    return fov_rad

def compute_footprint_corners(row, ground_altitude, HFOV, VFOV, target_crs):
    agl = row["altitude"] - ground_altitude
    if agl <= 0: agl = 10 
    
    # Calculate Grid Convergence
    transformer = Transformer.from_crs(target_crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(row['geometry'].x, row['geometry'].y)
    p_proj = Proj(target_crs)
    factors = p_proj.get_factors(lon, lat)
    convergence_deg = factors.meridian_convergence
    convergence_rad = np.radians(convergence_deg)
    
    # Corrected Yaw
    y = row['yaw'] - convergence_rad
    p, r = 0.0, row['roll']
    
    half_width = (agl * np.tan(HFOV/2)) / np.cos(r)
    half_height = (agl * np.tan(VFOV/2)) / np.cos(p)
    
    local_corners = np.array([
        [-half_width,  half_height],
        [ half_width,  half_height],
        [ half_width, -half_height],
        [-half_width, -half_height]
    ])
    
    cos_y = np.cos(-y)
    sin_y = np.sin(-y)
    R = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
    rotated_corners = local_corners @ R.T
    utm_corners = rotated_corners + np.array([row['geometry'].x, row['geometry'].y])
    return utm_corners

def load_geotiff(path):
    with rasterio.open(path) as src:
        img = src.read() # (bands, rows, cols)
        return img, src.profile

def get_8bit_nir(img_stack):
    # NIR band is typically 4th (index 3)
    band = img_stack[3, :, :]
    # Use 2th and 98th percentile for robust normalization
    p2, p98 = np.percentile(band, (2, 98))
    if p98 > p2:
        band_8bit = np.clip((band - p2) / (p98 - p2) * 255, 0, 255).astype('uint8')
    else:
        band_8bit = np.zeros(band.shape, dtype='uint8')
    return band_8bit

def get_homography(kp1, des1, kp2, des2):
    if des1 is None or des2 is None: return None
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < 10: return None
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H

def process_aoi(aoi_id, root_folder, ground_altitude, target_crs="EPSG:32616"):
    aoi_geojson = os.path.join(root_folder, f"Processed/AOI/aoi_{aoi_id}.geojson")
    if not os.path.exists(aoi_geojson): return
    gdf = gpd.read_file(aoi_geojson)
    if len(gdf) < 2: return

    tiff_path = os.path.join(root_folder, "Processed/capture")
    save_path = os.path.join(root_folder, f"Processed/stitched/aoi_{aoi_id}")
    os.makedirs(save_path, exist_ok=True)

    images_full = []
    images_8bit = []
    footprints = []

    print(f"AOI {aoi_id}: Loading images for Bundle Adjustment...")
    for _, row in gdf.iterrows():
        p = os.path.join(tiff_path, f"{row['image_name']}.tif")
        img, _ = load_geotiff(p)
        images_full.append(img)
        
        b8 = get_8bit_nir(img)
        # Stitcher needs 3-channel images
        images_8bit.append(cv2.merge([b8, b8, b8]))
        
        m = metadata.Metadata(p)
        w, h = m.image_size()
        HFOV = calculate_fov(m.get_item("EXIF:FocalLength"), w)
        VFOV = calculate_fov(m.get_item("EXIF:FocalLength"), h)
        footprints.append(compute_footprint_corners(row, ground_altitude, HFOV, VFOV, target_crs))

    print(f"AOI {aoi_id}: Running OpenCV Global Bundle Adjustment (Stitcher)...")
    stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
    status, result_ba_8bit = stitcher.stitch(images_8bit)

    if status != cv2.Stitcher_OK:
        print(f"Bundle Adjustment failed for AOI {aoi_id} (Status: {status}).")
        return

    # To georeference the BA result, we map all bands of all images into the result's coordinate space.
    # The stitcher provides a mosaic, but we need to warp our multi-band data to match it.
    # Easiest way: Match the first image to the result to "anchor" the whole thing.
    sift = cv2.SIFT_create()
    kp_ref, des_ref = sift.detectAndCompute(images_8bit[0], None)
    kp_res, des_res = sift.detectAndCompute(result_ba_8bit, None)
    
    H_ref_to_res = get_homography(kp_ref, des_ref, kp_res, des_res)
    if H_ref_to_res is None:
        print(f"Could not anchor BA result to UTM for AOI {aoi_id}")
        return

    # Now we need to warp ALL bands. 
    # OpenCV Stitcher doesn't easily return individual homographies in python, 
    # but we can warp our original multiband images to match the BA result.
    num_bands = images_full[0].shape[0]
    res_h, res_w = result_ba_8bit.shape[0], result_ba_8bit.shape[1]
    mosaic_multiband = np.zeros((num_bands, res_h, res_w), dtype=images_full[0].dtype)

    print(f"AOI {aoi_id}: Warping all bands to match BA result...")
    for i in range(len(images_full)):
        kp_curr, des_curr = sift.detectAndCompute(images_8bit[i], None)
        H_curr_to_res = get_homography(kp_curr, des_curr, kp_res, des_res)
        
        if H_curr_to_res is not None:
            for b in range(num_bands):
                warped = cv2.warpPerspective(images_full[i][b], H_curr_to_res, (res_w, res_h))
                mask = (warped > 0)
                mosaic_multiband[b][mask] = warped[mask]

    # Georeference using the anchored first footprint
    fp = footprints[0]
    w_img, h_img = images_full[0].shape[2], images_full[0].shape[1]
    img_corners = np.float32([[0, 0], [w_img, 0], [w_img, h_img], [0, h_img]]).reshape(-1, 1, 2)
    res_corners = cv2.perspectiveTransform(img_corners, H_ref_to_res).reshape(-1, 2)

    gcp_list = [
        GroundControlPoint(row=res_corners[0,1], col=res_corners[0,0], x=fp[0,0], y=fp[0,1]),
        GroundControlPoint(row=res_corners[1,1], col=res_corners[1,0], x=fp[1,0], y=fp[1,1]),
        GroundControlPoint(row=res_corners[2,1], col=res_corners[2,0], x=fp[2,0], y=fp[2,1]),
        GroundControlPoint(row=res_corners[3,1], col=res_corners[3,0], x=fp[3,0], y=fp[3,1])
    ]

    transform = from_gcps(gcp_list)
    output_meta = {
        'driver': 'GTiff',
        'dtype': mosaic_multiband.dtype,
        'nodata': 0,
        'width': res_w,
        'height': res_h,
        'count': num_bands,
        'crs': target_crs,
        'transform': transform,
        'compress': 'lzw'
    }

    output_file = os.path.join(save_path, f"aoi_{aoi_id}_stitched.tif")
    with rasterio.open(output_file, 'w', **output_meta) as dst:
        dst.write(mosaic_multiband)
    print(f"Saved BA-refined mosaic: {output_file}")

if __name__ == "__main__":
    root_folder = "091425_Wallpe"
    panel_dir = os.path.join(root_folder, "Panel")
    panel_file = next(Path(panel_dir).glob("*.tif"))
    ground_alt = metadata.Metadata(panel_file.as_posix()).position()[2]
    target_crs = "EPSG:32616"
    
    for aoi in tqdm(range(24), desc="Processing All AOIs"):
        process_aoi(aoi, root_folder, ground_alt, target_crs)
