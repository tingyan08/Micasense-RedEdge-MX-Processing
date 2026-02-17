import cv2, os, json
import numpy as np
import geopandas as gpd
import micasense.metadata as metadata
from pathlib import Path
from scipy.spatial import KDTree
from osgeo import gdal
from tqdm import tqdm


def calculate_gsd(folder):
    img_dir = Path(folder) / "Images"
    panel_dir = Path(folder) / "Panel"
    img = next(Path(img_dir).glob("*.tif"))  # Assuming all images have the same altitude
    panel = next(Path(panel_dir).glob("*.tif"))  # Assuming all panels have the same altitude


    # Calculate AGL (Above Ground Level) using the altitude from the image and panel metadata
    img_meta = metadata.Metadata(img.resolve().as_posix())
    panel_meta = metadata.Metadata(panel.resolve().as_posix())
    image_alt = img_meta.position()[2]
    panel_alt = panel_meta.position()[2]
    agl = image_alt - panel_alt # AGL in meters
    agl = agl * 100  # Convert AGL to centimeters for GSD calculation
    print(f"Calculated AGL: {agl:.2f} cm")

    # Get aligned image metadata to confirm altitude after processing
    aligned_file = next((Path(folder) / "Processed" / "capture").glob("*.tif"))
    aligned_meta = metadata.Metadata(aligned_file.resolve().as_posix())
    # Sensor width and focal length 
    pixel_size = 1 / aligned_meta.get_item("EXIF:XResolution")  # in mm/pixel
    print(f"Pixel size: {pixel_size:.4f} mm/pixel")
    focal_length = aligned_meta.get_item("EXIF:FocalLength")  # in mm
    print(f"Focal length: {focal_length:.4f} mm")
    gsd = agl * pixel_size / focal_length  # GSD in mm/pixel

    return gsd

def load_geotiff(path):
    ds = gdal.Open(path)
    if ds is None:
        raise IOError(f"GDAL could not open {path}")

    rows = ds.RasterYSize
    cols = ds.RasterXSize
    bands_count = ds.RasterCount
    
    # Initialize an empty array with shape (rows, cols, bands)
    img_stack = np.zeros((rows, cols, bands_count), dtype=np.ushort) # 16-bit unsigned integer data
    band_descriptions = []

    for i in range(1, bands_count + 1):
        band = ds.GetRasterBand(i)
        # Read the array - GDAL reads as (rows, cols)
        img_stack[:, :, i-1] = band.ReadAsArray()
        band_descriptions.append(band.GetDescription())

    ds = None # Close dataset
    return img_stack, band_descriptions


def get_features(path, sift):
    # Check path exists
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    # Load 16-bit multi-band image
    img_stack, band_desc = load_geotiff(path)

    if img_stack is None:
        raise ValueError(f"Could not load image at {path}")

    # Use Band 4 (NIR) or Band 3 (Red) for feature extraction (usually high contrast)
    # ['Blue', 'Green', 'Red', 'NIR', 'Red edge']
    # OpenCV expects 8-bit for SIFT, so we normalize the specific band
    target_ch = "NIR"
    target_ch_index = band_desc.index(target_ch)
    band_to_match = img_stack[:, :, target_ch_index] 
    # print(f"Range of {target_ch} band before normalization:", np.min(band_to_match), np.max(band_to_match))
    # Since the image preprocessing part clip reflectance into the range of [0, 2], so 200% = 65535 and 100% = 32768 in 16-bit. We can scale it down to 0-255 for SIFT.
    band_8bit = np.clip(band_to_match / 32768 * 255, 0, 255).astype('uint8')
    # print(f"Range of {target_ch} band after normalization:", np.min(band_8bit), np.max(band_8bit))
    kp, des = sift.detectAndCompute(band_8bit, None)
    return kp, des, img_stack


def roi_blendering(mosaic, warped):
    # Find bounding box of non-zero pixels in the warped image
    pts = np.argwhere(np.any(warped > 0, axis=2))
    if pts.size > 0:
        y_min, x_min = pts.min(axis=0)
        y_max, x_max = pts.max(axis=0)
        roi_canvas = mosaic[y_min:y_max+1, x_min:x_max+1]
        roi_new = warped[y_min:y_max+1, x_min:x_max+1]
        mask = np.any(roi_new > 0, axis=2)
        roi_canvas[mask] = roi_new[mask]
    return mosaic

if __name__ == "__main__":
    # Define paths
    root_folder = "B1"
    image_folder = os.path.join(root_folder, "Processed", "capture")
    geojason_file = os.path.join(root_folder, "Processed", "imageSet.json")
    UTM_CRS = "EPSG:32616" # UTM Zone 16N for Indiana

    gdf = gpd.read_file(geojason_file)
    gdf['img_path'] = gdf['capture_id'].apply(lambda x: os.path.normpath(os.path.join(image_folder, f"{x}.tif")))
    gdf_utm = gdf.to_crs(UTM_CRS)
    GSD = calculate_gsd(root_folder) / 100  # Convert cm/pixel to m/pixel for geospatial calculations

    minx, miny, maxx, maxy = gdf_utm.total_bounds
    padding = 1000
    origin_x = minx - (padding * GSD)
    origin_y = maxy + (padding * GSD) 

    W2P = np.array([
        [1/GSD, 0, -origin_x/GSD],
        [0, -1/GSD, origin_y/GSD], 
        [0, 0, 1]
    ], dtype=np.float32)

    canvas_w = int((maxx - minx) / GSD) + 2000
    canvas_h = int((maxy - miny) / GSD) + 2000

    mosaic_ms = np.zeros((canvas_h, canvas_w, 5), dtype=np.uint16)
    mosaic_rgb = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Extract features for all images
    global_transforms = {}
    feature_cache = {}

    mosaic_path = os.path.join(root_folder, "Mosaic")
    if not os.path.exists(mosaic_path):
        os.makedirs(mosaic_path)

    # RGB visualization of the mosaic
    rgb_path = os.path.join(mosaic_path, "RGB")
    if not os.path.exists(rgb_path):
        os.makedirs(rgb_path)


    for i in tqdm(range(len(gdf)), desc="GPS Stacking"):
        stack_curr, _ = load_geotiff(gdf.iloc[i]['img_path'])
        if stack_curr is None: continue
        h, w = stack_curr.shape[:2]

        # GPS Transformation
        T_gps = np.array([[1, 0, gdf_utm.iloc[i].geometry.x], 
                          [0, 1, gdf_utm.iloc[i].geometry.y], 
                          [0, 0, 1]])
        T_center = np.array([[1, 0, -w/2], [0, 1, -h/2], [0, 0, 1]])

        M_global = W2P @ T_gps @ T_center

        # # Stacking 16-bit Multispectral
        # warped_ms = cv2.warpPerspective(stack_curr, M_global, (canvas_w, canvas_h))
        # mosaic_ms = roi_blendering(mosaic_ms, warped_ms)


        # E. Stacking 8-bit RGB Visualization
        # Micasense RedEdge-MX indices: R=2, G=1, B=0
        rgb_8bit = np.clip(stack_curr[:, :, [2, 1, 0]] / 32768 * 255, 0, 255).astype('uint8')
        warped_rgb = cv2.warpPerspective(rgb_8bit, M_global, (canvas_w, canvas_h))
        mosaic_rgb = roi_blendering(mosaic_rgb, warped_rgb)

        # Save rgb steps
        rgb_output_path = os.path.join(rgb_path, f"mosaic_step_{i:04d}.jpg")
        cv2.imwrite(rgb_output_path, mosaic_rgb)

    # # Save final multispectral mosaic as GeoTIFF
    # mosaic_out = os.path.join(root_folder, "Mosaic", "final_mosaic.tif")
    # driver = gdal.GetDriverByName('GTiff')
    # ds = driver.Create(mosaic_out, canvas_w, canvas_h, 5, gdal.GDT_UInt16)
    
    # # Update origin for GDAL based on the pixel offset applied
    # ds.SetGeoTransform((minx, GSD, 0, maxy, 0, -GSD))
    # for b in range(5):
    #     ds.GetRasterBand(b+1).WriteArray(mosaic_ms[:, :, b])
    # ds.SetProjection(gdf_utm.crs.to_wkt())
    # ds = None
    # print(f"Saved final multispectral mosaic to: {mosaic_out}")


    # Save final RGB mosaic for visualization
    mosaic_rgb = cv2.normalize(mosaic_rgb, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    rgb_mosaic_output_path = os.path.join(mosaic_path, "final_mosaic_rgb.jpg")
    cv2.imwrite(rgb_mosaic_output_path, mosaic_rgb)
    print(f"Saved final RGB mosaic to: {rgb_mosaic_output_path}")

        


    