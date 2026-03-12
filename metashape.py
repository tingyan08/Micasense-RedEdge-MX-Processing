import Metashape, os, time
import pandas as pd
import geopandas as gpd
import numpy as np
from pyproj import Transformer
from aoi_filtering import get_joined_gdf, calculate_characteristics

def metashape_pipeline(doc, images, panels, target_crs, chunk_label=None):
    chunk = doc.addChunk()
    if chunk_label:
        chunk.label = chunk_label

    # Add survey images and panel images
    chunk.addPhotos(images)
    chunk.addPhotos(panels)

    # 1. Align Photos
    # (0- Highest, 1- High, 2- Medium, 4- Low, 8- Lowest)
    chunk.matchPhotos(downscale=1, generic_preselection=True, reference_preselection=True)
    chunk.alignCameras()

    # 2. Define the Coordinate System
    crs = Metashape.CoordinateSystem(target_crs)

    # 3. Reflectance Calibration
    chunk.locateReflectancePanels()
    chunk.calibrateReflectance(use_reflectance_panels=True, use_sun_sensor=True)

    # 4. Camera Optimization
    chunk.optimizeCameras(fit_f=True, fit_cx=True, fit_cy=True, fit_b1=False, fit_b2=False, fit_k1=True,
                          fit_k2=True, fit_k3=True, fit_k4=False, fit_p1=True, fit_p2=True)

    # 5. Point Cloud (Depth Maps first, then Dense Cloud)
    # (1- Ultra high, 2- High, 4- Medium, 8- Low, 16- Lowest)
    chunk.buildDepthMaps(downscale=2, filter_mode=Metashape.AggressiveFiltering)
    chunk.buildPointCloud(
        source_data=Metashape.DepthMapsData,
        point_colors=True,
        point_confidence=True,
        keep_depth=True
    )

    # 6. DEM (Digital Elevation Model)
    ortho_projection = Metashape.OrthoProjection()
    ortho_projection.crs = crs
    chunk.buildDem(
        source_data=Metashape.PointCloudData, 
        interpolation=Metashape.EnabledInterpolation,
        projection=ortho_projection,
    )

    # 7. Orthomosaic
    chunk.buildOrthomosaic(
        surface_data=Metashape.ElevationData,
        blending_mode=Metashape.MosaicBlending, # Mosaic (default)
        fill_holes=True,                        # Enable hole filling
        refine_seamlines=False,                 # Default is off
        ghosting_filter=False,                  # Default is off
        projection=ortho_projection
    )
    doc.save()
    print(f"Pipeline complete for {chunk.label}")

if __name__ == "__main__":
    parent_folder = "Data"
    exp = "091425_Wallpe"
    root_folder = os.path.join(parent_folder, exp)
    aoi_id = 5

    gsd_cm, H, W = calculate_characteristics(root_folder)
    width_m = gsd_cm / 100 * W
    height_m = gsd_cm / 100 * H 
    print(f"Calculated GSD: {gsd_cm:.2f} cm/pixel")
    print(f"Image dimensions: {H} x {W} pixels")

    original_crs = "EPSG:4326"  # WGS 84
    target_crs = "EPSG:32616"  # WGS 84 - UTM zone 16N
    transformer = Transformer.from_crs(original_crs, target_crs, always_xy=True)

    aoi_df = pd.read_csv(os.path.join(root_folder, "aoi.csv"))
    capture_gdf = gpd.read_file(os.path.join(root_folder, "Processed/imageSet.json"))
    capture_gdf = capture_gdf.to_crs(target_crs)

    recorded_info = {"ratio": [], "num_captures": [], "process_time": []}
    for ratio in np.arange(0.05, 0.51, 0.05):
        joined_gdf = get_joined_gdf(aoi_df, capture_gdf, width_m, height_m, transformer, target_crs, ratio=ratio, aoi_id=aoi_id, aoi_size=36)
        print(f"There are {len(joined_gdf)} captures in the AOI {aoi_id} with ratio {ratio}.")

        doc = Metashape.Document()
        try:
            doc.open(os.path.join(root_folder, "Metashape", f"{exp}.psx"))
        except:
            doc.save(os.path.join(root_folder, "Metashape", f"{exp}.psx"))
        images = [os.path.join(root_folder, "Images", f"{i}_{j}.tif") for i in joined_gdf['image_name'].tolist() for j in range(1, 6)]
        panels = [os.path.join(root_folder, "Panel", f"IMG_0000_{i}.tif") for i in range(1, 6)]

        start_time = time.time()
        metashape_pipeline(doc, images, panels, target_crs, chunk_label=f"AOI_{aoi_id}_ratio_{ratio:.2f}")
        process_time = time.time() - start_time
        recorded_info["ratio"].append(ratio)
        recorded_info["num_captures"].append(len(joined_gdf))
        recorded_info["process_time"].append(process_time)

    results_df = pd.DataFrame(recorded_info)
    results_df.to_csv(os.path.join(root_folder, "Metashape", f"aoi_{aoi_id}", "processing_results.csv"), index=False)
