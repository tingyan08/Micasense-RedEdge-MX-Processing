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
    
    # 1. Calculate Grid Convergence
    # We need to convert UTM back to Lat/Lon to get convergence
    transformer = Transformer.from_crs(target_crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(row['geometry'].x, row['geometry'].y)
    
    # Get convergence angle at this specific location
    p_proj = Proj(target_crs)
    factors = p_proj.get_factors(lon, lat)
    convergence_deg = factors.meridian_convergence # Angle between True North and Grid North
    convergence_rad = np.radians(convergence_deg)
    
    # 2. Apply Correction: Corrected_Yaw = True_Yaw - Convergence
    y = row['yaw'] - convergence_rad
    p, r = row['pitch'], row['roll']
    
    # Scaling: stretch footprint based on tilt
    half_width = (agl * np.tan(HFOV/2)) / np.cos(r)
    half_height = (agl * np.tan(VFOV/2)) / np.cos(p)
    
    # Corner offsets in local (unrotated) meters
    local_corners = np.array([
        [-half_width,  half_height], # Top Left
        [ half_width,  half_height], # Top Right
        [ half_width, -half_height], # Bottom Right
        [-half_width, -half_height]  # Bottom Left
    ])
    
    # 3. Rotation: Corrected heading
    cos_y = np.cos(-y)
    sin_y = np.sin(-y)
    R = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
    
    rotated_corners = local_corners @ R.T
    
    # Translate to UTM
    utm_corners = rotated_corners + np.array([row['geometry'].x, row['geometry'].y])
    return utm_corners

def load_geotiff(path):
    with rasterio.open(path) as src:
        img = src.read() # (bands, rows, cols)
        return img, src.profile

def get_sift_features(img_stack):
    # NIR band is typically 4th (index 3)
    band_to_match = img_stack[3, :, :]
    # Normalize for SIFT (Assuming 16-bit where 32768 is ~100% reflectance)
    # Using a robust normalization to handle both individual frames and merged mosaics
    band_min, band_max = band_to_match.min(), band_to_match.max()
    if band_max > band_min:
        band_8bit = np.clip((band_to_match - band_min) / (band_max - band_min) * 255, 0, 255).astype('uint8')
    else:
        band_8bit = np.zeros(band_to_match.shape, dtype='uint8')
        
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(band_8bit, None)
    return kp, des, band_8bit

def get_homography(kp1, des1, kp2, des2, img1_8bit=None, img2_8bit=None, save_path=None):
    if des1 is None or des2 is None: return None
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if save_path and img1_8bit is not None and img2_8bit is not None:
        match_img = cv2.drawMatches(img1_8bit, kp1, img2_8bit, kp2, good, None, 
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(save_path, match_img)

    if len(good) < 10: return None
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H

def warp_images_multiband(img_ref, img_to_warp, H, ref_gcps, warp_gcps):
    """
    Logic from image_stiching.py warpImages, adapted for multiband (C, H, W) and GCPs.
    img_ref: The "reference" image (pasted on top)
    img_to_warp: The image being warped by H
    H: Homography from img_to_warp to img_ref
    """
    c, h1, w1 = img_ref.shape
    _, h2, w2 = img_to_warp.shape

    # Corners
    pts_ref = np.float32([[0,0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    pts_warp = np.float32([[0,0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    
    transformed_pts_warp = cv2.perspectiveTransform(pts_warp, H)
    all_pts = np.concatenate((pts_ref, transformed_pts_warp), axis=0)

    x_min, y_min = np.int32(all_pts.min(axis=0).ravel() - 0.5)
    x_max, y_max = np.int32(all_pts.max(axis=0).ravel() + 0.5)
    
    tx, ty = -x_min, -y_min
    H_translation = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
    
    new_w, new_h = x_max - x_min, y_max - y_min
    
    # Warp img_to_warp
    warped_img = np.zeros((c, new_h, new_w), dtype=img_ref.dtype)
    for b in range(c):
        warped_img[b] = cv2.warpPerspective(img_to_warp[b], H_translation @ H, (new_w, new_h))
    
    # Paste img_ref on top (follows image_stiching.py logic)
    warped_img[:, ty:h1+ty, tx:w1+tx] = img_ref
    
    # Update GCPs
    new_gcps = []
    # Reference image GCPs just need translation
    for gcp in ref_gcps:
        new_gcps.append(GroundControlPoint(row=gcp.row + ty, col=gcp.col + tx, x=gcp.x, y=gcp.y))
    
    # Warped image GCPs need transformation then translation
    for gcp in warp_gcps:
        p = np.float32([[[gcp.col, gcp.row]]])
        new_p = cv2.perspectiveTransform(p, H_translation @ H).reshape(-1)
        new_gcps.append(GroundControlPoint(row=new_p[1], col=new_p[0], x=gcp.x, y=gcp.y))
        
    return warped_img, new_gcps

def process_aoi(aoi_id, root_folder, ground_altitude, target_crs="EPSG:32616"):
    aoi_geojson = os.path.join(root_folder, f"Processed/AOI/aoi_{aoi_id}.geojson")
    gdf = gpd.read_file(aoi_geojson)
    tiff_path = os.path.join(root_folder, "Processed/capture")
    
    # Prepare directory for matching results
    
    # Initial items list: each item is {'img': array, 'gcps': [GroundControlPoint]}
    items = []

    save_path = os.path.join(root_folder, f"Processed/stitched/aoi_{aoi_id}")
    os.makedirs(save_path, exist_ok=True)

    print(f"Processing AOI {aoi_id}: Loading images and metadata...")
    for _, row in gdf.iterrows():
        img_name = row['image_name']
        p = os.path.join(tiff_path, f"{img_name}.tif")

        img, _ = load_geotiff(p)
        m = metadata.Metadata(p)
        focal_length = m.get_item("EXIF:FocalLength")
        w_img, h_img = m.image_size()

        HFOV = calculate_fov(focal_length, w_img)
        VFOV = calculate_fov(focal_length, h_img)
        fp_corners = compute_footprint_corners(row, ground_altitude, HFOV, VFOV, target_crs)

        # Create initial GCPs for the 4 corners of this frame
        item_gcps = [
            GroundControlPoint(row=0, col=0, x=fp_corners[0,0], y=fp_corners[0,1]),
            GroundControlPoint(row=0, col=w_img, x=fp_corners[1,0], y=fp_corners[1,1]),
            GroundControlPoint(row=h_img, col=w_img, x=fp_corners[2,0], y=fp_corners[2,1]),
            GroundControlPoint(row=h_img, col=0, x=fp_corners[3,0], y=fp_corners[3,1])
        ]
        items.append({'img': img, 'gcps': item_gcps})

    # Iterative stitching logic from image_stiching.py
    idx = 1
    while len(items) > 1:
        item1 = items.pop(0) # Previous mosaic result
        item2 = items.pop(0) # Next image in sequence
        
        # SIFT and Matching (re-computing on mosaic at each step)
        kp1, des1, b8_1 = get_sift_features(item1['img'])
        kp2, des2, b8_2 = get_sift_features(item2['img'])
        
        save_match_path = os.path.join(save_path, f"matches_{idx}.jpg")
        # H maps item1 to item2
        H = get_homography(kp1, des1, kp2, des2, b8_1, b8_2, save_match_path)
        
        if H is None:
            print(f"Match failed for AOI {aoi_id} at step {idx}. Skipping this pair.")
            # Put the next image back as the start of a new potential mosaic
            items.insert(0, item2)
            idx += 1
            continue
            
        # Warp item1 (prev) into item2 (next) space and paste item2 on top
        new_img, new_gcps = warp_images_multiband(item2['img'], item1['img'], H, item2['gcps'], item1['gcps'])
        
        items.insert(0, {'img': new_img, 'gcps': new_gcps})
        idx += 1

    if not items: return
    
    final_item = items[0]
    mosaic = final_item['img']
    gcp_list = final_item['gcps']

    # Define output metadata for rasterio using GCPs
    transform = from_gcps(gcp_list)
    output_meta = {
        'driver': 'GTiff',
        'dtype': mosaic.dtype,
        'nodata': 0,
        'width': mosaic.shape[2],
        'height': mosaic.shape[1],
        'count': mosaic.shape[0],
        'crs': target_crs,
        'transform': transform,
        'compress': 'lzw'
    }

    
    output_file = os.path.join(save_path, f"aoi_{aoi_id}_stitched.tif")
    
    with rasterio.open(output_file, 'w', **output_meta) as dst:
        dst.write(mosaic)
    print(f"Saved georeferenced mosaic: {output_file}")

if __name__ == "__main__":
    root_folder = "091425_Wallpe"
    # Dynamic ground altitude from Panel metadata
    panel_dir = os.path.join(root_folder, "Panel")
    panel_file = next(Path(panel_dir).glob("*.tif"))
    ground_alt = metadata.Metadata(panel_file.as_posix()).position()[2]
    target_crs = "EPSG:32616" # UTM Zone 16N
    
    for aoi in tqdm(range(24), desc="Processing All AOIs"):
        process_aoi(aoi, root_folder, ground_alt, target_crs)