import os, glob
import json
from shlex import join
import pandas as pd
import geopandas as gpd
from micasense import metadata
from pathlib import Path
from pyproj import Transformer
from shapely.geometry import Point, box
from tqdm import tqdm


def calculate_characteristics(root_folder):
    img_dir = Path(root_folder) / "Images"
    panel_dir = Path(root_folder) / "Panel"
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
    aligned_file = next((Path(root_folder) / "Processed" / "capture").glob("*.tif"))
    aligned_meta = metadata.Metadata(aligned_file.resolve().as_posix())
    # Sensor width and focal length 
    pixel_size = 1 / aligned_meta.get_item("EXIF:XResolution")  # in mm/pixel
    print(f"Pixel size: {pixel_size:.4f} mm/pixel")
    focal_length = aligned_meta.get_item("EXIF:FocalLength")  # in mm
    print(f"Focal length: {focal_length:.4f} mm")
    gsd = agl * pixel_size / focal_length  # GSD in cm/pixel

    H = aligned_meta.get_item("EXIF:ImageHeight")
    W = aligned_meta.get_item("EXIF:ImageWidth")

    return gsd, H, W



def get_joined_gdf(aoi_df, capture_gdf,width_m, height_m, crs_transformer, target_crs, ratio=0.5, aoi_id=0, aoi_size=36):
    # Expand the aoi 
    selected_polygon = []
    t_east, t_north = crs_transformer.transform(aoi_df.iloc[aoi_id]['Longitude'], aoi_df.iloc[aoi_id]['Latitude'])
    expanded_poly = box(t_east - aoi_size/2 - (0.5 - ratio) * width_m, t_north - aoi_size/2 -(0.5 - ratio) * height_m, 
                        t_east + aoi_size/2 + (0.5 - ratio) * width_m, t_north + aoi_size/2 + (0.5 - ratio) * height_m)
    selected_polygon.append({"id": 0, "polygon": expanded_poly, "center": (t_east, t_north)})
    
    selected_gdf = gpd.GeoDataFrame(pd.DataFrame(selected_polygon), geometry='polygon', crs=target_crs)
    joined_gdf = gpd.sjoin(capture_gdf,  selected_gdf.iloc[[0]], how="inner", predicate="within")
    joined_gdf = joined_gdf.drop(columns=["index_right", 'id'])

    return joined_gdf

if __name__ == "__main__":
    root_folder = "091425_Wallpe"
    gsd_cm, H, W = calculate_characteristics(root_folder)
    width_m = gsd_cm / 100 * W
    height_m = gsd_cm / 100 * H
    print(f"Calculated GSD: {gsd_cm:.2f} cm/pixel")
    print(f"Image dimensions: {H} x {W} pixels")

    original_crs = "EPSG:4326"  # WGS 84
    target_crs = "EPSG:32616"  # WGS 84 - UTM zone 16N
    transformer = Transformer.from_crs(original_crs, target_crs, always_xy=True)

    # Area of interest
    area_of_interest = pd.read_csv(os.path.join(root_folder, "aoi.csv"))
    aoi_polygon = []
    aoi_size = 36
    for i, row in area_of_interest.iterrows():
        t_east, t_north = transformer.transform(row['Longitude'], row['Latitude'])
        t_poly = box(t_east - aoi_size/2, t_north - aoi_size/2, t_east + aoi_size/2, t_north + aoi_size/2)
        aoi_polygon.append({"id": i, "polygon": t_poly, "center": (t_east, t_north)})

    # Expand the aoi 
    ratio = 0.5
    selected_polygon = []
    for i, row in area_of_interest.iterrows():
        t_east, t_north = transformer.transform(row['Longitude'], row['Latitude'])
        t_poly = box(t_east - aoi_size/2, t_north - aoi_size/2, t_east + aoi_size/2, t_north + aoi_size/2)
        expanded_poly = box(t_east - aoi_size/2 - (0.5 - ratio) * width_m, t_north - aoi_size/2 -(0.5 - ratio) * height_m, 
                            t_east + aoi_size/2 + (0.5 - ratio) * width_m, t_north + aoi_size/2 + (0.5 - ratio) * height_m)
        selected_polygon.append({"id": i, "polygon": expanded_poly, "center": (t_east, t_north)})
    
    selected_gdf = gpd.GeoDataFrame(pd.DataFrame(selected_polygon), geometry='polygon', crs=target_crs)



    save_path = os.path.join(root_folder, "Processed", "AOI")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    capture_gdf = gpd.read_file(os.path.join(root_folder, "Processed/imageSet.json"), crs=original_crs)
    capture_gdf = capture_gdf.to_crs(target_crs)
    for i, row in selected_gdf.iterrows():
        print(f"AOI {row['id']}")
        joined_gdf = gpd.sjoin(capture_gdf,  selected_gdf.iloc[[i]], how="inner", predicate="within")
        joined_gdf = joined_gdf.drop(columns=["index_right", 'id'])
        print(f"Found {len(joined_gdf)} captures in AOI {row['id']}")
        joined_gdf.to_file(os.path.join(save_path, f"aoi_{row['id']}.geojson"), driver="GeoJSON")



    
