import Metashape, os
import pandas as pd
import geopandas as gpd
from micasense.imageset import ImageSet

class ImageSet(ImageSet):
    def as_nested_lists(self):
        """
        Get timestamp, latitude, longitude, altitude, capture_id, dls-yaw, dls-pitch, dls-roll, and irradiance from all
        Captures.
        :return: List data from all Captures, List column headers.
        """
        columns = [
            'timestamp',
            'latitude', 'longitude', 'altitude',
            'capture_id',
            'image_name',
            'dls-yaw', 'dls-pitch', 'dls-roll'
        ]
        irr = ["irr-{}".format(wve) for wve in self.captures[0].center_wavelengths()]
        columns += irr
        data = []
        for cap in self.captures:
            dat = cap.utc_time()
            loc = list(cap.location())
            uuid = cap.uuid
            img_name = cap.images[0].path.split("/")[-1].split(".")[0][:-2]
            dls_pose = list(cap.dls_pose())
            irr = cap.dls_irradiance()
            row = [dat] + loc + [uuid] + [img_name] + dls_pose + irr
            data.append(row)
        return data, columns




def metashape_pipeline(result_folder, doc, images, panels, target_crs, chunk_label=None):
    chunk = doc.addChunk()
    if chunk_label:
        chunk.label = chunk_label

    # Add survey images and panel images
    chunk.addPhotos(images)
    chunk.addPhotos(panels)
    doc.save()

    # 1. Reflectance Calibration
    chunk.locateReflectancePanels()
    chunk.calibrateReflectance(use_reflectance_panels=True, use_sun_sensor=True)
    doc.save()

    # 2. Align Photos
    # (0- Highest, 1- High, 2- Medium, 4- Low, 8- Lowest)
    chunk.matchPhotos(
        downscale=1, 
        generic_preselection=True, 
        reference_preselection=True, 
        filter_stationary_points=True,
        reference_preselection_mode=Metashape.ReferencePreselectionSource,
    )
    chunk.alignCameras()
    doc.save()


    # 3. Define the Coordinate System
    crs = Metashape.CoordinateSystem(target_crs)

    # 4. Camera Optimization
    chunk.optimizeCameras(fit_f=True, fit_cx=True, fit_cy=True, fit_b1=False, fit_b2=False, fit_k1=True,
                          fit_k2=True, fit_k3=True, fit_k4=False, fit_p1=True, fit_p2=True)
    doc.save()


    # 5. Point Cloud (Depth Maps first, then Dense Cloud)
    # (1- Ultra high, 2- High, 4- Medium, 8- Low, 16- Lowest)
    chunk.buildDepthMaps(downscale=2, filter_mode=Metashape.AggressiveFiltering)
    doc.save()

    chunk.buildPointCloud(
        source_data=Metashape.DepthMapsData,
        point_confidence=True,
        point_colors=True,
        keep_depth=True
    )
    doc.save()

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

    # 8. Export the orthomosaic
    ortho_output_dir = os.path.join(result_folder, "Orthomosaics")
    if not os.path.exists(ortho_output_dir):
        os.makedirs(ortho_output_dir)
    if chunk.orthomosaic:
        orthomosaic_path = os.path.join(ortho_output_dir, f"{chunk.label}_orthomosaic.tif")
        
        chunk.exportRaster(
            path=orthomosaic_path,
            format=Metashape.RasterFormatTiles, # Standard for large GeoTIFFs
            image_format=Metashape.ImageFormatTIFF,
            projection=ortho_projection,
            save_alpha=True
        )
        print(f"Orthomosaic exported to: {orthomosaic_path}")
    else:
        print(f"Error: No orthomosaic found in chunk {chunk.label}. Build it first.")

    # 9. Export Individual Orthophotos ---
    rectified_output_dir = os.path.join(result_folder, "Orthorectified", chunk.label)
    if not os.path.exists(rectified_output_dir):
        os.makedirs(rectified_output_dir)
    chunk.exportOrthophotos(
        path=os.path.join(rectified_output_dir, "{filename}.tif"),
        projection=ortho_projection,
    )
    print(f"Individual orthophotos exported to: {rectified_output_dir}")

    
    # Export a PDF report for the chunk
    if not os.path.exists(os.path.join(result_folder, "report")):
        os.makedirs(os.path.join(result_folder, "report"))
    chunk.exportReport(path=os.path.join(result_folder, "report", f"{chunk.label}_report.pdf"))
    doc.save()
    print(f"Pipeline complete for {chunk.label}")

def get_capture_gdf(root_folder):
    if not os.path.exists(os.path.join(root_folder, "imageSet.geojson")):
        print("imageSet.geojson not found, generating from Images directory...")
        img_dir = os.path.join(root_folder, "Images")
        imgset = ImageSet.from_directory(img_dir)
        data, columns = imgset.as_nested_lists()
        capture_gdf = pd.DataFrame.from_records(data, index='timestamp', columns=columns)
        capture_gdf = gpd.GeoDataFrame(capture_gdf, geometry=gpd.points_from_xy(capture_gdf.longitude, capture_gdf.latitude), crs="EPSG:4326")
        capture_gdf.to_file(os.path.join(root_folder, "imageSet.geojson"), driver="GeoJSON")
    else:
        capture_gdf = gpd.read_file(os.path.join(root_folder, "imageSet.geojson"))
    return capture_gdf
