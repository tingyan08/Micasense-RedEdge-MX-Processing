import os, sys, math, warnings, time
import cv2 as cv
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_gcps
from rasterio.control import GroundControlPoint
from tqdm import tqdm
from pyproj import Proj, Transformer
import micasense.metadata as metadata
from pathlib import Path

# Suppress Rasterio warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# --- CONFIGURATION ---
TARGET_CRS = "EPSG:32616"
WORK_MEGAPIX = 0.6    # Resolution for matching only. Increase if matching fails.
CONF_THRESH = 1.0     # Confidence threshold for matching images.
BLEND_STRENGTH = 5    # Blending strength [0-100].
# ---------------------

def calculate_fov(focal_length_mm, pixels, pixel_size_mm=0.00375):
    sensor_size_mm = pixels * pixel_size_mm
    fov_rad = 2 * math.atan(sensor_size_mm / (2 * focal_length_mm))
    return fov_rad

def compute_footprint_corners(row, ground_altitude, HFOV, VFOV, target_crs):
    agl = row["altitude"] - ground_altitude
    if agl <= 0: agl = 10 
    
    transformer = Transformer.from_crs(target_crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(row['geometry'].x, row['geometry'].y)
    p_proj = Proj(target_crs)
    factors = p_proj.get_factors(lon, lat)
    convergence_rad = np.radians(factors.meridian_convergence)
    
    y = row['yaw'] - convergence_rad
    p, r = row['pitch'], row['roll']
    
    half_width = (agl * np.tan(HFOV/2)) / np.cos(r)
    half_height = (agl * np.tan(VFOV/2)) / np.cos(p)
    
    local_corners = np.array([
        [-half_width,  half_height],
        [ half_width,  half_height],
        [ half_width, -half_height],
        [-half_width, -half_height]
    ])
    
    cos_y, sin_y = np.cos(-y), np.sin(-y)
    R = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
    rotated_corners = local_corners @ R.T
    return rotated_corners + np.array([row['geometry'].x, row['geometry'].y])


def process_aoi_detailed(aoi_id, root_folder, ground_altitude):
    total_start = time.perf_counter()

    def log_step(step_name, step_start):
        elapsed = time.perf_counter() - step_start
        print(f"AOI {aoi_id}: {step_name} completed in {elapsed:.2f}s")

    aoi_geojson = os.path.join(root_folder, f"Processed/AOI/aoi_{aoi_id}.geojson")
    if not os.path.exists(aoi_geojson): return
    gdf = gpd.read_file(aoi_geojson)
    if len(gdf) < 2: return

    tiff_path = os.path.join(root_folder, "Processed/capture")
    rgb_path = os.path.join(root_folder, "Processed/thumbnails")
    save_path = os.path.join(root_folder, f"Processed/stitched_detailed")
    os.makedirs(save_path, exist_ok=True)

    features, images_8bit, images_full, footprints, full_img_sizes, utm_centers = [], [], [], [], [], []
    finder = cv.SIFT_create()

    print(f"AOI {aoi_id}: Loading data...")
    step_start = time.perf_counter()
    current_work_scale = 1.0
    for _, row in gdf.iterrows():
        name = row['image_name']
        p_tif = os.path.join(tiff_path, f"{name}.tif")
        p_rgb = os.path.join(rgb_path, f"{name}.jpg")
        
        with rasterio.open(p_tif) as src:
            img_stack = src.read()
            images_full.append(img_stack)
            full_img_sizes.append((img_stack.shape[2], img_stack.shape[1]))
        
        # Store GPS UTM center
        utm_centers.append((row['geometry'].x, row['geometry'].y))
        
        img_rgb = cv.imread(p_rgb)
        images_8bit.append(img_rgb)
        
        # Registration scale
        current_work_scale = min(1.0, np.sqrt(WORK_MEGAPIX * 1e6 / (img_rgb.shape[0] * img_rgb.shape[1])))
        img_work = cv.resize(img_rgb, None, fx=current_work_scale, fy=current_work_scale, interpolation=cv.INTER_LINEAR_EXACT)
        features.append(cv.detail.computeImageFeatures2(finder, img_work))

        m = metadata.Metadata(p_tif)
        HFOV = calculate_fov(m.get_item("EXIF:FocalLength"), img_stack.shape[2])
        VFOV = calculate_fov(m.get_item("EXIF:FocalLength"), img_stack.shape[1])
        footprints.append(compute_footprint_corners(row, ground_altitude, HFOV, VFOV, TARGET_CRS))
    log_step("Loading data", step_start)

    # 2. Matching and Bundle Adjustment
    step_start = time.perf_counter()
    matcher = cv.detail_BestOf2NearestMatcher(False, 0.3)
    p_matches = matcher.apply2(features)
    matcher.collectGarbage()

    indices = cv.detail.leaveBiggestComponent(features, p_matches, CONF_THRESH)
    if len(indices) < 2:
        print(f"AOI {aoi_id}: Not enough matches.")
        return

    features = [features[i] for i in indices]
    images_8bit = [images_8bit[i] for i in indices]
    images_full = [images_full[i] for i in indices]
    footprints = [footprints[i] for i in indices]
    full_img_sizes = [full_img_sizes[i] for i in indices]
    utm_centers = [utm_centers[i] for i in indices]

    estimator = cv.detail_HomographyBasedEstimator()
    success, cameras = estimator.apply(features, p_matches, None)
    if not success: return

    for cam in cameras:
        cam.R = cam.R.astype(np.float32)
        cam.t = cam.t.astype(np.float32)

    adjuster = cv.detail_BundleAdjusterRay()
    adjuster.setConfThresh(CONF_THRESH)
    success, cameras = adjuster.apply(features, p_matches, cameras)
    if not success: return
    log_step("Matching and bundle adjustment", step_start)

    # Scale parameters back to full resolution
    compose_scale = 1.0 / current_work_scale
    for cam in cameras:
        cam.focal *= compose_scale
        cam.ppx *= compose_scale
        cam.ppy *= compose_scale

    # 3. Warping
    step_start = time.perf_counter()
    focals = sorted([cam.focal for cam in cameras])
    warped_image_scale = focals[len(focals) // 2]
    warper = cv.PyRotationWarper('plane', warped_image_scale)

    corners, sizes = [], []
    for i in range(len(cameras)):
        sz = (full_img_sizes[i][0], full_img_sizes[i][1])
        roi = warper.warpRoi(sz, cameras[i].K().astype(np.float32), cameras[i].R)
        corners.append(roi[0:2])
        sizes.append(roi[2:4])

    dst_sz = cv.detail.resultRoi(corners=corners, sizes=sizes)
    log_step("Warping setup", step_start)

    # 4. Seam Finding (8-bit)
    step_start = time.perf_counter()
    images_warped, masks_warped = [], []
    for i in range(len(cameras)):
        K = cameras[i].K().astype(np.float32)
        _, img_wp = warper.warp(images_8bit[i], K, cameras[i].R, cv.INTER_LINEAR, cv.BORDER_REFLECT)
        images_warped.append(img_wp)
        mask = 255 * np.ones((images_8bit[i].shape[0], images_8bit[i].shape[1]), np.uint8)
        _, mask_wp = warper.warp(mask, K, cameras[i].R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)
        masks_warped.append(mask_wp)

    seam_finder = cv.detail.SeamFinder_createDefault(cv.detail.SeamFinder_VORONOI_SEAM)
    images_warped_f = [img.astype(np.float32) for img in images_warped]
    masks_warped = seam_finder.find(images_warped_f, corners, masks_warped)
    log_step("Seam finding", step_start)

    # 5. Composition (Multi-band)
    step_start = time.perf_counter()
    num_bands = images_full[0].shape[0]
    final_mosaic = np.zeros((num_bands, dst_sz[3], dst_sz[2]), dtype=images_full[0].dtype)

    print(f"AOI {aoi_id}: Blending {num_bands} bands at full resolution...")
    for b_idx in range(num_bands):
        blender = cv.detail_MultiBandBlender()
        blend_width = np.sqrt(dst_sz[2] * dst_sz[3]) * BLEND_STRENGTH / 100
        blender.setNumBands((np.log(blend_width) / np.log(2.) - 1.).astype(np.int32))
        blender.prepare(dst_sz)

        for i in range(len(cameras)):
            K = cameras[i].K().astype(np.float32)
            _, band_wp = warper.warp(images_full[i][b_idx], K, cameras[i].R, cv.INTER_LINEAR, cv.BORDER_REFLECT)
            band_wp_3ch = cv.merge([band_wp, band_wp, band_wp])
            blender.feed(band_wp_3ch.astype(np.int16), masks_warped[i], corners[i])
        
        res, _ = blender.blend(None, None)
        final_mosaic[b_idx] = np.clip(res[:, :, 0], 0, 65535).astype(images_full[0].dtype)
    log_step("Composition and blending", step_start)

    # 6. Georeferencing (Multi-center GCPs)
    print(f"AOI {aoi_id}: Generating GCPs from image centers...")
    step_start = time.perf_counter()
    gcp_list = []
    for i in range(len(cameras)):
        # Local center of image i
        local_center = (full_img_sizes[i][0] / 2, full_img_sizes[i][1] / 2)
        
        # Map center to mosaic pixel space
        K = cameras[i].K().astype(np.float32)
        pt_warped = warper.warpPoint(local_center, K, cameras[i].R)
        rel_x, rel_y = pt_warped[0] - dst_sz[0], pt_warped[1] - dst_sz[1]
        
        # UTM GPS coordinate
        utm_x, utm_y = utm_centers[i]
        gcp_list.append(GroundControlPoint(row=rel_y, col=rel_x, x=utm_x, y=utm_y))    
    
    output_file = os.path.join(save_path, f"aoi_{aoi_id}_detailed.tif")
    with rasterio.open(output_file, 'w', driver='GTiff', height=dst_sz[3], width=dst_sz[2],
                        count=num_bands, dtype=images_full[0].dtype, crs=TARGET_CRS, 
                        transform=from_gcps(gcp_list)) as dst:
        dst.write(final_mosaic)
    log_step("Georeferencing and save", step_start)
    print(f"AOI {aoi_id}: Total processing time {time.perf_counter() - total_start:.2f}s")
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    root_folder = "091425_Wallpe"
    aoi = 5
    panel_dir = os.path.join(root_folder, "Panel")
    panel_file = next(Path(panel_dir).glob("*.tif"))
    ground_alt = metadata.Metadata(panel_file.as_posix()).position()[2]
    process_aoi_detailed(aoi, root_folder, ground_alt)
