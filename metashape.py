import Metashape, os
import pandas as pd
import geopandas as gpd
import micasense.imageset as imageset

def metashape_pipeline(root_folder, doc, images, panels, target_crs, chunk_label=None):
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
    chunk.exportReport(path=os.path.join(root_folder, "Metashape", f"{chunk.label}_report.pdf"))
    doc.save()
    print(f"Pipeline complete for {chunk.label}")

def get_capture_gdf(root_folder):
    if not os.path.exists(os.path.join(root_folder, "imageSet.geojson")):
        print("imageSet.geojson not found, generating from Images directory...")
        img_dir = os.path.join(root_folder, "Images")
        imgset = imageset.ImageSet.from_directory(img_dir)
        data, columns = imgset.as_nested_lists()
        capture_gdf = pd.DataFrame.from_records(data, index='timestamp', columns=columns)
        capture_gdf = gpd.GeoDataFrame(capture_gdf, geometry=gpd.points_from_xy(capture_gdf.longitude, capture_gdf.latitude), crs="EPSG:4326")
        capture_gdf.to_file(os.path.join(root_folder, "imageSet.geojson"), driver="GeoJSON")
    else:
        capture_gdf = gpd.read_file(os.path.join(root_folder, "imageSet.geojson"))
    return capture_gdf
