import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.mask
import Metashape
from pyproj import Transformer


def compute_vi(vi_name, b1, b2, b3, b4, b5):
    """Compute a vegetation index from reflectance bands (0–1 scale)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        if vi_name == "NDVI":
            return (b5 - b3) / (b5 + b3)
        elif vi_name == "GNDVI":
            return (b5 - b2) / (b5 + b2)
        elif vi_name == "SAVI":
            return 1.5 * (b5 - b3) / (b5 + b3 + 0.5)
        elif vi_name == "NDRE":
            return (b5 - b4) / (b5 + b4)
        elif vi_name == "TVI":
            return 0.5 * (120 * (b5 - b2) - 200 * (b3 - b2))
        elif vi_name == "ExG":
            return 2 * b2 - b3 - b1
        elif vi_name == "SR":
            return np.where(b3 > 0, b5 / b3, np.nan)
        elif vi_name == "PSRI":
            return np.where(b5 > 0, (b3 - b2) / b5, np.nan)
        elif vi_name == "G":
            return np.where(b3 > 0, b2 / b3, np.nan)
        elif vi_name == "RDVI":
            denom = np.sqrt(np.maximum(b5 + b3, 0))
            return np.where(denom > 0, (b5 - b3) / denom, np.nan)
        elif vi_name == "MCARI2":
            num = 1.5 * (2.5 * (b5 - b3) - 1.3 * (b5 - b2))
            inner = (2 * b5 + 1) ** 2 - (6 * b5 - 5 * np.sqrt(np.maximum(b5, 0))) - 0.5
            denom = np.sqrt(np.maximum(inner, 0))
            return np.where(denom > 0, num / denom, np.nan)
        elif vi_name == "GRVI":
            return (b2 - b3) / (b2 + b3)
        elif vi_name == "Red":
            return b3
        elif vi_name == "Green":
            return b2
        elif vi_name == "Blue":
            return b1
        elif vi_name == "RedEdge":
            return b4
        elif vi_name == "NIR":
            return b5
        elif vi_name == "OSAVI":
            return 1.16 * (b5 - b3) / (b5 + b3 + 0.16)
        else:
            raise ValueError(f"Unknown VI: {vi_name}")


if __name__ == "__main__":
    parent_folder = "Data/Wallpe"
    # exps = ["080625_Wallpe", "081325_Wallpe", "081525_Wallpe", "082525_Wallpe", "083025_Wallpe", "091025_Wallpe", "091425_Wallpe"]
    exps = ["082525_Wallpe"]
    aoi_file = "wallpe_aoi_square.geojson"


    for exp in exps:
        root_folder = os.path.join(parent_folder, exp)
        result_folder = os.path.join(root_folder, "Metashape", "AOI_results")
        vi_folder = os.path.join(result_folder, "Vegetation_Indices")
        vi_masked_folder = os.path.join(result_folder, "Vegetation_Indices_masked")
        ortho_folder = os.path.join(result_folder, "Orthomosaics")

        # Band order for MicaSense RedEdge-M: B1=Blue, B2=Green, B3=Red, B4=RedEdge, B5=NIR
        vi_names = ["NDVI", "GNDVI", "SAVI", "NDRE", "TVI", "ExG", "SR", "PSRI", "G", "RDVI", "MCARI2", "GRVI", "Red", "Green", "Blue", "RedEdge", "NIR", "OSAVI"]


        # Soil/background removal thresholds (applied using reflectance-normalized bands).
        # Pixels are removed (set to NaN) if HSV 'value' < hsv_v_threshold OR RDVI < rdvi_threshold.
        hsv_v_threshold = 0.05   # HSV value = max(R, G, B) in 0–1 reflectance space
        rdvi_threshold  = 0.10   # RDVI threshold

        utm_crs = "EPSG:32616"
        wgs84_crs = "EPSG:4326"
        transformer = Transformer.from_crs(utm_crs, wgs84_crs, always_xy=True)

        os.makedirs(vi_folder, exist_ok=True)
        os.makedirs(vi_masked_folder, exist_ok=True)
        os.makedirs(ortho_folder, exist_ok=True)

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

        records_vi_masked = []
        records_vi_raw = []

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
                row_nan = {"aoi_id": aoi_id, "latitude": lat, "longitude": lon, "canopy_cover": np.nan,
                    "total_pixels": 0, "vegetation_pixels": 0}
                for vi_name in vi_names:
                    vi_key = vi_name.lower()
                    row_nan[f"{vi_key}_mean"] = np.nan
                    row_nan[f"{vi_key}_std"]  = np.nan
                    row_nan[f"{vi_key}_min"]  = np.nan
                    row_nan[f"{vi_key}_max"]  = np.nan
                records_vi_masked.append(row_nan)
                records_vi_raw.append(row_nan)
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

            # --- Load orthomosaic, mask to polygon, normalize to reflectance ---
            with rasterio.open(ortho_path) as src:
                ortho_masked, crop_transform = rasterio.mask.mask(src, [polygon], crop=True, all_touched=True)
                ortho_nodata = src.nodata
                vi_profile = src.profile.copy()

            vi_profile.update(count=1, dtype="float32", nodata=np.nan, transform=crop_transform,
                            width=ortho_masked.shape[2], height=ortho_masked.shape[1])

            b1 = ortho_masked[0].astype(np.float64) / 32768  # Blue
            b2 = ortho_masked[1].astype(np.float64) / 32768  # Green
            b3 = ortho_masked[2].astype(np.float64) / 32768  # Red
            b4 = ortho_masked[3].astype(np.float64) / 32768  # RedEdge
            b5 = ortho_masked[4].astype(np.float64) / 32768  # NIR

            # Mark nodata pixels across all bands
            nodata_mask = np.zeros_like(b1, dtype=bool)
            if ortho_nodata is not None:
                for band in ortho_masked:
                    nodata_mask |= (band == ortho_nodata)

            # HSV 'value' = max(R, G, B) in reflectance space
            hsv_v = np.maximum(np.maximum(b3, b2), b1)

            # RDVI for soil removal (reflectance scale, denominator same units)
            denom = np.sqrt(np.maximum(b5 + b3, 0))
            rdvi_inline = np.where(denom > 0, (b5 - b3) / denom, np.nan)

            print(f"  RDVI range: {np.nanmin(rdvi_inline):.4f} – {np.nanmax(rdvi_inline):.4f}")

            # Vegetation mask: keep pixels passing both thresholds
            veg_mask = (
                (hsv_v >= hsv_v_threshold) &
                (rdvi_inline >= rdvi_threshold) &
                ~nodata_mask
            )
            total_valid = int((~nodata_mask).sum())
            print(f"  Vegetation pixels after soil removal: {int(veg_mask.sum())} / {total_valid}")


            row = {"aoi_id": aoi_id, "latitude": lat, "longitude": lon, "canopy_cover": float(veg_mask.sum()) / total_valid,
                "total_pixels": total_valid, "vegetation_pixels": int(veg_mask.sum())}
            
            row_raw = {"aoi_id": aoi_id, "latitude": lat, "longitude": lon, "canopy_cover": 1,
                    "total_pixels": total_valid, "vegetation_pixels": total_valid}

            # --- Compute each VI from bands, save unmasked and masked GeoTIFFs ---
            # For the unmasked version, all valid pixels are included (soil pixels have NaN values).
            # For the masked version, soil/background pixels are set to NaN, so stats reflect only vegetation pixels.
            # Save both versions for potential future use, and export both stats to compare the impact of soil removal.
            for vi_name in vi_names:
                vi_path        = os.path.join(vi_folder,        f"{chunk.label}_{vi_name}.tif")
                vi_masked_path = os.path.join(vi_masked_folder, f"{chunk.label}_{vi_name}_masked.tif")

                data = compute_vi(vi_name, b1, b2, b3, b4, b5)
                data[nodata_mask] = np.nan

                # Unmasked: all valid pixels
                with rasterio.open(vi_path, "w", **vi_profile) as dst:
                    dst.write(data.astype(np.float32), 1)

                vi_key = vi_name.lower()
                if len(data[~nodata_mask]) > 0:
                    row_raw[f"{vi_key}_mean"] = float(np.nanmean(data))
                    row_raw[f"{vi_key}_std"]  = float(np.nanstd(data))
                    row_raw[f"{vi_key}_min"]  = float(np.nanmin(data))
                    row_raw[f"{vi_key}_max"]  = float(np.nanmax(data))
                else:
                    row_raw[f"{vi_key}_mean"] = np.nan
                    row_raw[f"{vi_key}_std"]  = np.nan
                    row_raw[f"{vi_key}_min"]  = np.nan
                    row_raw[f"{vi_key}_max"]  = np.nan

                

                # Masked: soil/background pixels set to NaN
                data_masked = data.copy()
                data_masked[~veg_mask] = np.nan
                with rasterio.open(vi_masked_path, "w", **vi_profile) as dst:
                    dst.write(data_masked.astype(np.float32), 1)

                if len(data_masked) > 0:
                    row[f"{vi_key}_mean"] = float(np.nanmean(data_masked))
                    row[f"{vi_key}_std"]  = float(np.nanstd(data_masked))
                    row[f"{vi_key}_min"]  = float(np.nanmin(data_masked))
                    row[f"{vi_key}_max"]  = float(np.nanmax(data_masked))
                else:
                    row[f"{vi_key}_mean"] = np.nan
                    row[f"{vi_key}_std"]  = np.nan
                    row[f"{vi_key}_min"]  = np.nan
                    row[f"{vi_key}_max"]  = np.nan

                print(f"  {vi_name}: mean={row[f'{vi_key}_mean']:.4f}  std={row[f'{vi_key}_std']:.4f}")

            records_vi_masked.append(row)
            records_vi_raw.append(row_raw)
            print()

        # Build and save summary dataframe
        df = pd.DataFrame(records_vi_masked).sort_values("aoi_id").reset_index(drop=True)
        out_csv = os.path.join("./features", f"{exp}_aoi_no_soil.csv")
        df.to_csv(out_csv, index=False)
        print(f"Saved VI stats -> {out_csv}")
        print("Done.")

        df_raw = pd.DataFrame(records_vi_raw).sort_values("aoi_id").reset_index(drop=True)
        out_csv_raw = os.path.join("./features", f"{exp}_aoi_raw.csv")
        df_raw.to_csv(out_csv_raw, index=False)
        print(f"Saved unmasked VI stats -> {out_csv_raw}")
        print("Done.")
