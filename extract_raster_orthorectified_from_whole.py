import glob
import os
from pydoc import doc
import Metashape
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from tqdm import tqdm
from pyproj import Transformer
from aoi_filtering import get_joined_gdf, calculate_characteristics
from metashape import get_capture_gdf




def _update_acc(acc: dict, values: np.ndarray) -> None:
    valid = values.ravel()
    valid = valid[~np.isnan(valid)]
    n_new = len(valid)
    if n_new == 0:
        return
    mean_new = float(valid.mean())
    M2_new = float(np.sum((valid - mean_new) ** 2))
    n_old = acc["n"]
    if n_old == 0:
        acc["n"], acc["mean"], acc["M2"] = n_new, mean_new, M2_new
        acc["min"], acc["max"] = float(valid.min()), float(valid.max())
    else:
        delta = mean_new - acc["mean"]
        n_total = n_old + n_new
        acc["mean"] = (n_old * acc["mean"] + n_new * mean_new) / n_total
        acc["M2"] += M2_new + delta ** 2 * n_old * n_new / n_total
        acc["n"] = n_total
        acc["min"] = min(acc["min"], float(valid.min()))
        acc["max"] = max(acc["max"], float(valid.max()))


def _acc_stats(acc: dict) -> tuple:
    n = acc["n"]
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan
    mean = acc["mean"]
    std  = np.sqrt(acc["M2"] / n) if n > 1 else 0.0
    return mean, std, acc["min"], acc["max"]


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
    field = "PPAC-B3"
    parent_folder = f"Data/{field}"
    if field == "Wallpe":
        exps = ["080625_Wallpe", "081325_Wallpe", "081525_Wallpe", "082525_Wallpe", "083025_Wallpe", "091025_Wallpe", "091425_Wallpe"]
    else:
        exps = ["061724_PPAC-B3", "071124_PPAC-B3", "072324_PPAC-B3", "080324_PPAC-B3", "081324_PPAC-B3", "081924_PPAC-B3", "090124_PPAC-B3", "090824_PPAC-B3"]
    aoi_file = f"{field}_aoi.csv"

    # Band order for MicaSense RedEdge-M: B1=Blue, B2=Green, B3=Red, B4=RedEdge, B5=NIR
    vi_names = ["NDVI", "GNDVI", "SAVI", "NDRE", "TVI", "ExG", "SR", "PSRI", "G", "RDVI", "MCARI2", "GRVI", "Red", "Green", "Blue", "RedEdge", "NIR", "OSAVI"]


    # Image selection threshold
    ratio = 0.05
    aoi_size = 12 if "PPAC-B3" in parent_folder else 36

    # Soil/background removal thresholds (applied using reflectance-normalized bands).
    # Pixels are removed (set to NaN) if HSV 'value' < hsv_v_threshold OR RDVI < rdvi_threshold.
    hsv_v_threshold = 0.05   # HSV value = max(R, G, B) in 0–1 reflectance space
    rdvi_threshold  = 0.15 if field == "PPAC-B3" else 0.10  # RDVI threshold

    utm_crs = "EPSG:32616"
    wgs84_crs = "EPSG:4326"
    utm2wgs_transformer = Transformer.from_crs(utm_crs, wgs84_crs, always_xy=True)
    wgs2utm_transformer = Transformer.from_crs(wgs84_crs, utm_crs, always_xy=True)

    for exp in exps:
        records_vi_masked = []
        records_vi_raw = []

        gsd_cm, H, W = calculate_characteristics(os.path.join(parent_folder, exp))
        width_m = gsd_cm / 100 * W
        height_m = gsd_cm / 100 * H 
        print(f"Calculated GSD: {gsd_cm:.2f} cm/pixel")
        print(f"Image dimensions: {H} x {W} pixels")

        aoi_df = pd.read_csv(os.path.join(parent_folder, aoi_file))
        capture_gdf = get_capture_gdf(os.path.join(parent_folder, exp))
        capture_gdf = capture_gdf.to_crs(utm_crs)

        # Check if the orthorectified folder exists or is empty for this experiment. If not, try to export the orthophotos for the whole field using the Metashape project, then proceed with VI extraction.
        orthorectified_folder = os.path.join(parent_folder, exp, "Metashape", "whole_field", "Orthorectified")
        if not os.path.exists(orthorectified_folder) or not os.listdir(orthorectified_folder):
            print(f"Orthorectified folder not found for {exp}. Attempting to export orthophotos from Metashape project...")
            metashape_project = os.path.join(parent_folder, exp, "Metashape", "whole_field", f"{exp}_whole_field.psx")
            if os.path.exists(metashape_project):
                doc = Metashape.Document()
                doc.open(metashape_project)
                chunk = doc.chunk
                export_path = os.path.join(parent_folder, exp, "Metashape", "whole_field", "Orthorectified")
                ortho_projection = Metashape.OrthoProjection()
                os.makedirs(export_path, exist_ok=True)
                chunk.exportOrthophotos(
                    path=os.path.join(export_path, "{filename}.tif"),
                    projection=ortho_projection,
                )
                print(f"Orthophotos exported to {export_path}.")
                print("Export complete. Proceeding with VI extraction...\n")
            else:
                print(f"Error: Metashape project not found at {metashape_project}. Cannot export orthophotos. Skipping VI extraction for {exp}.\n")
                continue

        else:
            print(f"Orthorectified folder already exists for {exp}. Proceeding with VI extraction...\n")


        
        
        aoi_range = aoi_df['Point_ID'].unique()
        

        for aoi_id in sorted(aoi_range):
            # For each AOI, find all orthophotos that intersect it, compute VIs, and aggregate stats.
            joined_gdf = get_joined_gdf(aoi_df, capture_gdf, width_m, height_m, wgs2utm_transformer, utm_crs, ratio=ratio, aoi_id=aoi_id, aoi_size=aoi_size)
            lon, lat = aoi_df[aoi_df['Point_ID'] == aoi_id][['Longitude', 'Latitude']].values[0]

            print(f"Processing AOI {aoi_id}...")
            
            row = {"aoi_id": aoi_id, "latitude": lat, "longitude": lon, "canopy_cover": 0,
                "total_pixels": 0, "vegetation_pixels": 0}
            
            row_raw = {"aoi_id": aoi_id, "latitude": lat, "longitude": lon, "canopy_cover": 1,
                    "total_pixels": 0, "vegetation_pixels": 0}
            
            acc_raw    = {vi: {"n": 0, "mean": 0.0, "M2": 0.0, "min": np.inf, "max": -np.inf} for vi in vi_names}
            acc_masked = {vi: {"n": 0, "mean": 0.0, "M2": 0.0, "min": np.inf, "max": -np.inf} for vi in vi_names}

            # Scan all orthophotos for this AOI (there may be multiple if the AOI overlaps multiple chunks), compute VIs, and aggregate stats.
            for image_name in tqdm(joined_gdf['image_name'].tolist(), desc=f"  Processing orthophotos for AOI {aoi_id}", unit="photo"):
                orth = os.path.join(parent_folder, exp, "Metashape", "whole_field", "Orthorectified", f"{image_name}_1.tif")
                # --- Load orthomosaic, mask to polygon, normalize to reflectance ---
                with rasterio.open(orth) as src:
                    ortho = src.read()  # (bands, rows, cols)
                    ortho_nodata = src.nodata
                    ortho_meta = src.meta

                b1 = ortho[0].astype(np.float64) / 32768  # Blue
                b2 = ortho[1].astype(np.float64) / 32768  # Green
                b3 = ortho[2].astype(np.float64) / 32768  # Red
                b4 = ortho[3].astype(np.float64) / 32768  # RedEdge
                b5 = ortho[4].astype(np.float64) / 32768  # NIR

                # Mark nodata pixels across all bands
                nodata_mask = np.zeros_like(b1, dtype=bool)
                if ortho_nodata is not None:
                    for band in ortho:
                        nodata_mask |= (band == ortho_nodata)

                # HSV 'value' = max(R, G, B) in reflectance space
                hsv_v = np.maximum(np.maximum(b3, b2), b1)

                # RDVI for soil removal (reflectance scale, denominator same units)
                denom = np.sqrt(np.maximum(b5 + b3, 0))
                with np.errstate(divide="ignore", invalid="ignore"):
                    rdvi_inline = np.where(denom > 0, (b5 - b3) / denom, np.nan)

                # print(f"  RDVI range: {np.nanmin(rdvi_inline):.4f} – {np.nanmax(rdvi_inline):.4f}")

                # Vegetation mask: keep pixels passing both thresholds
                veg_mask = (
                    (hsv_v >= hsv_v_threshold) &
                    (rdvi_inline >= rdvi_threshold) &
                    ~nodata_mask
                )
                total_valid = int((~nodata_mask).sum())
                # print(f"  Vegetation pixels after soil removal: {int(veg_mask.sum())} / {total_valid}")

                row["total_pixels"] += total_valid
                row["vegetation_pixels"] += int(veg_mask.sum())
                row["canopy_cover"] = row["vegetation_pixels"] / row["total_pixels"] if row["total_pixels"] > 0 else 0

                for vi_name in vi_names:
                    data = compute_vi(vi_name, b1, b2, b3, b4, b5)
                    data[nodata_mask] = np.nan
                    _update_acc(acc_raw[vi_name], data)

                    data_masked = data.copy()
                    data_masked[~veg_mask] = np.nan
                    _update_acc(acc_masked[vi_name], data_masked)

                row_raw["total_pixels"] = row["total_pixels"]
                row_raw["vegetation_pixels"] = row["total_pixels"]

            # --- Compute stats for this AOI across all orthophotos ---
            for vi_name in vi_names:
                vi_key = vi_name.lower()
                row[f"{vi_key}_mean"], row[f"{vi_key}_std"], row[f"{vi_key}_min"], row[f"{vi_key}_max"] = _acc_stats(acc_masked[vi_name])
                row_raw[f"{vi_key}_mean"], row_raw[f"{vi_key}_std"], row_raw[f"{vi_key}_min"], row_raw[f"{vi_key}_max"] = _acc_stats(acc_raw[vi_name])

    
            records_vi_masked.append(row)
            records_vi_raw.append(row_raw)
            print()

        # Build and save summary dataframe
        df = pd.DataFrame(records_vi_masked).sort_values("aoi_id").reset_index(drop=True)
        out_csv = os.path.join(f"./features/{field}", f"{exp}_orthorectified_no_soil.csv")
        df.to_csv(out_csv, index=False)
        print(f"Saved VI stats -> {out_csv}")
        print("Done.")

        df_raw = pd.DataFrame(records_vi_raw).sort_values("aoi_id").reset_index(drop=True)
        out_csv_raw = os.path.join(f"./features/{field}", f"{exp}_orthorectified_raw.csv")
        df_raw.to_csv(out_csv_raw, index=False)
        print(f"Saved unmasked VI stats -> {out_csv_raw}")
        print("Done.")
