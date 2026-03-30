import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.mask
import Metashape
from pyproj import Transformer


if __name__ == "__main__":
    parent_folder = "Data"
    exp = "091425_Wallpe"
    aoi_file = "wallpe_aoi_square.geojson"

    root_folder = os.path.join(parent_folder, exp)
    result_folder = os.path.join(root_folder, "Metashape", "AOI_results")
    vi_folder = os.path.join(result_folder, "Vegetation_Indices")
    ortho_folder = os.path.join(result_folder, "Orthomosaics")

    # Band order for MicaSense RedEdge-M: B1=Blue, B2=Green, B3=Red, B4=RedEdge, B5=NIR
    vi_formulas = {
        "NDVI":  "(B5 - B3) / (B5 + B3)",
        "GNDVI": "(B5 - B2) / (B5 + B2)",
        "CIRE":  "(B5 / B4) - 1",
        "MCARI": "((B4 - B3) - 0.2 * (B4 - B2)) / 32768 * (B4 / B3) ",
        "NDRE":  "(B5 - B4) / (B5 + B4)",
        "SAVI":  "1.5 * (B5 / 32768 - B3 / 32768) / (B5 / 32768 + B3 / 32768 + 0.5)", # Assume L=0.5 for SAVI
    }

    utm_crs = "EPSG:32616"
    wgs84_crs = "EPSG:4326"
    transformer = Transformer.from_crs(utm_crs, wgs84_crs, always_xy=True)

    os.makedirs(vi_folder, exist_ok=True)

    # Load AOI polygons, indexed by id
    aoi_gdf = gpd.read_file(os.path.join(parent_folder, aoi_file))
    aoi_gdf = aoi_gdf.set_index("id")

    # Open Metashape project
    psx_path = os.path.join(result_folder, f"{exp}_aoi.psx")
    doc = Metashape.Document()
    try:
        doc.open(psx_path)
    except Exception:
        raise FileNotFoundError(
            f"Project file not found at {psx_path}. "
            "Please run metashape_process_all_aoi.py first."
        )

    print(f"Opened project: {psx_path}")
    print(f"Found {len(doc.chunks)} chunk(s).\n")

    records = []

    for chunk in doc.chunks:
        # Parse AOI id from label, e.g. "AOI_5_ratio_0.05" -> 5
        parts = chunk.label.split("_")
        try:
            aoi_id = int(parts[1])
        except (IndexError, ValueError):
            print(f"  Could not parse AOI id from label '{chunk.label}', skipping.")
            continue

        if chunk.orthomosaic is None:
            print(f"  Skipping {chunk.label}: no orthomosaic found.\n")
            continue

        if aoi_id not in aoi_gdf.index:
            print(f"  AOI id {aoi_id} not found in GeoJSON, skipping.\n")
            continue

        polygon = aoi_gdf.loc[aoi_id, "geometry"]
        xmin, ymin, xmax, ymax = polygon.bounds

        # Clip region for Metashape export
        region = Metashape.BBox()
        region.min = Metashape.Vector([xmin, ymin])
        region.max = Metashape.Vector([xmax, ymax])

        # Center coordinates in WGS84
        lon, lat = transformer.transform(polygon.centroid.x, polygon.centroid.y)

        print(f"Processing chunk: {chunk.label}  (AOI {aoi_id})")

        # --- Export 5-band orthomosaic clipped to AOI ---
        ortho_path = os.path.join(ortho_folder, f"{chunk.label}.tif")
        chunk.exportRaster(
            path=ortho_path,
            source_data=Metashape.OrthomosaicData,
            image_format=Metashape.ImageFormatTIFF,
            split_in_blocks=False,
            save_alpha=False,
            region=region,
        )
        print(f"  Exported 5-band orthomosaic -> {ortho_path}")

        row = {"aoi_id": aoi_id, "latitude": lat, "longitude": lon}

        # --- Export each VI and compute masked stats ---
        for vi_name, formula in vi_formulas.items():
            vi_path = os.path.join(vi_folder, f"{chunk.label}_{vi_name}.tif")

            
            chunk.raster_transform.formula = [formula]
            chunk.raster_transform.calibrateRange()
            chunk.raster_transform.enabled = True


            chunk.exportRaster(
                path=vi_path,
                source_data=Metashape.OrthomosaicData,
                raster_transform=Metashape.RasterTransformValue,
                image_format=Metashape.ImageFormatTIFF,
                split_in_blocks=False,
                save_alpha=False,
                region=region,
            )

            # Mask to exact polygon boundary and compute stats
            with rasterio.open(vi_path) as src:
                masked, _ = rasterio.mask.mask(src, [polygon], crop=True, all_touched=True)
                data = masked[0].astype(np.float64)
                nodata_val = src.nodata
                if nodata_val is not None:
                    data[data == nodata_val] = np.nan

            valid = data[np.isfinite(data)]
            vi_key = vi_name.lower()
            if len(valid) > 0:
                row[f"{vi_key}_mean"] = float(np.mean(valid))
                row[f"{vi_key}_std"]  = float(np.std(valid))
                row[f"{vi_key}_min"]  = float(np.min(valid))
                row[f"{vi_key}_max"]  = float(np.max(valid))
            else:
                row[f"{vi_key}_mean"] = np.nan
                row[f"{vi_key}_std"]  = np.nan
                row[f"{vi_key}_min"]  = np.nan
                row[f"{vi_key}_max"]  = np.nan

            print(f"  {vi_name}: mean={row[f'{vi_key}_mean']:.4f}  std={row[f'{vi_key}_std']:.4f}")

        records.append(row)
        print()

    # Build and save summary dataframe
    df = pd.DataFrame(records).sort_values("aoi_id").reset_index(drop=True)
    out_csv = os.path.join(result_folder, f"{exp}_vi_stats.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved VI stats -> {out_csv}")
    print("Done.")
