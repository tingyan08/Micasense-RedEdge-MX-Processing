import glob
import os
import Metashape
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import box
from tqdm import tqdm
from pyproj import Transformer




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


def get_intersecting_aois(bounds, ortho_crs, aoi_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return the AOI squares (reprojected to `ortho_crs`) whose bounding boxes intersect `bounds`."""
    aoi_proj = aoi_gdf.to_crs(ortho_crs)
    bbox_gdf = gpd.GeoDataFrame(geometry=[box(*bounds)], crs=ortho_crs)
    return gpd.sjoin(aoi_proj, bbox_gdf, how="inner", predicate="intersects")


def get_aoi_pixel_masks(intersecting: gpd.GeoDataFrame, transform, out_shape: tuple) -> dict:
    """For each AOI square, return a boolean pixel mask (True = pixel falls inside that AOI square)."""
    masks = {}
    for aoi_id, aoi_row in intersecting.iterrows():
        mask = geometry_mask([aoi_row.geometry], out_shape=out_shape, transform=transform, invert=True)
        if mask.any():
            masks[aoi_id] = mask
    return masks


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


def ensure_orthophotos_exported(parent_folder: str, exp: str) -> str | None:
    """Return the path to the whole-field Orthorectified folder for `exp`, exporting it from the
    Metashape project first if it doesn't exist yet. Returns None if it cannot be made available."""
    orthorectified_folder = os.path.join(parent_folder, exp, "Metashape", "whole_field", "Orthorectified")
    if os.path.exists(orthorectified_folder) and os.listdir(orthorectified_folder):
        print(f"Orthorectified folder already exists for {exp}. Proceeding with VI extraction...\n")
        return orthorectified_folder

    print(f"Orthorectified folder not found for {exp}. Attempting to export orthophotos from Metashape project...")
    metashape_project = os.path.join(parent_folder, exp, "Metashape", "whole_field", f"{exp}_whole_field.psx")
    if not os.path.exists(metashape_project):
        print(f"Error: Metashape project not found at {metashape_project}. Cannot export orthophotos. Skipping VI extraction for {exp}.\n")
        return None

    doc = Metashape.Document()
    doc.open(metashape_project)
    chunk = doc.chunk
    ortho_projection = Metashape.OrthoProjection()
    os.makedirs(orthorectified_folder, exist_ok=True)
    chunk.exportOrthophotos(
        path=os.path.join(orthorectified_folder, "{filename}.tif"),
        projection=ortho_projection,
    )
    print(f"Orthophotos exported to {orthorectified_folder}.")
    print("Export complete. Proceeding with VI extraction...\n")
    return orthorectified_folder


def init_aoi_accumulators(aoi_square_gdf: gpd.GeoDataFrame, vi_names: list, utm2wgs_transformer: Transformer) -> tuple:
    """Initialize the per-AOI output rows and VI accumulators for one experiment."""
    rows, rows_raw, acc_raw, acc_masked = {}, {}, {}, {}
    for aoi_id, aoi_row in aoi_square_gdf.iterrows():
        centroid = aoi_row.geometry.centroid
        lon, lat = utm2wgs_transformer.transform(centroid.x, centroid.y)
        rows[aoi_id] = {"aoi_id": aoi_id, "latitude": lat, "longitude": lon, "canopy_cover": 0,
                        "total_pixels": 0, "vegetation_pixels": 0}
        rows_raw[aoi_id] = {"aoi_id": aoi_id, "latitude": lat, "longitude": lon, "canopy_cover": 1,
                            "total_pixels": 0, "vegetation_pixels": 0}
        acc_raw[aoi_id] = {vi: {"n": 0, "mean": 0.0, "M2": 0.0, "min": np.inf, "max": -np.inf} for vi in vi_names}
        acc_masked[aoi_id] = {vi: {"n": 0, "mean": 0.0, "M2": 0.0, "min": np.inf, "max": -np.inf} for vi in vi_names}
    return rows, rows_raw, acc_raw, acc_masked


def process_image(orth_path: str, aoi_square_gdf: gpd.GeoDataFrame, vi_names: list,
                   hsv_v_threshold: float, rdvi_threshold: float,
                   rows: dict, rows_raw: dict, acc_raw: dict, acc_masked: dict) -> None:
    """Find the AOI squares that `orth_path` intersects, mask the image to each one, and accumulate
    raw + vegetation-masked VI stats into `acc_raw`/`acc_masked` (and pixel counts into `rows`/`rows_raw`)."""
    with rasterio.open(orth_path) as src:
        bounds = src.bounds
        transform = src.transform
        ortho_crs = src.crs

        intersecting = get_intersecting_aois(bounds, ortho_crs, aoi_square_gdf)
        if intersecting.empty:
            return  # No AOI squares intersect this orthophoto

        ortho = src.read()  # (bands, rows, cols)
        ortho_nodata = src.nodata

    b1 = ortho[0].astype(np.float64) / 32768  # Blue
    b2 = ortho[1].astype(np.float64) / 32768  # Green
    b3 = ortho[2].astype(np.float64) / 32768  # Red
    b4 = ortho[3].astype(np.float64) / 32768  # RedEdge
    b5 = ortho[4].astype(np.float64) / 32768  # NIR

    # Mark nodata pixels across all bands. If a 6th (alpha) band is present, use it directly.
    if ortho.shape[0] >= 6:
        nodata_mask = ortho[5] == 0
    else:
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

    # Vegetation mask: keep pixels passing both thresholds
    veg_mask = (
        (hsv_v >= hsv_v_threshold) &
        (rdvi_inline >= rdvi_threshold) &
        ~nodata_mask
    )

    # Pre-compute every VI for the whole image once (re-used across all AOIs it overlaps)
    vi_data = {}
    for vi_name in vi_names:
        data = compute_vi(vi_name, b1, b2, b3, b4, b5)
        data[nodata_mask] = np.nan
        vi_data[vi_name] = data

    out_shape = ortho.shape[1:]
    aoi_pixel_masks = get_aoi_pixel_masks(intersecting, transform, out_shape)
    for aoi_id, poly_mask in aoi_pixel_masks.items():
        valid_mask = poly_mask & ~nodata_mask
        veg_in_poly = poly_mask & veg_mask

        rows[aoi_id]["total_pixels"] += int(valid_mask.sum())
        rows[aoi_id]["vegetation_pixels"] += int(veg_in_poly.sum())
        rows[aoi_id]["canopy_cover"] = (
            rows[aoi_id]["vegetation_pixels"] / rows[aoi_id]["total_pixels"]
            if rows[aoi_id]["total_pixels"] > 0 else 0
        )
        rows_raw[aoi_id]["total_pixels"] = rows[aoi_id]["total_pixels"]
        rows_raw[aoi_id]["vegetation_pixels"] = rows[aoi_id]["total_pixels"]

        for vi_name in vi_names:
            data_in_poly = np.where(poly_mask, vi_data[vi_name], np.nan)
            _update_acc(acc_raw[aoi_id][vi_name], data_in_poly)

            data_masked = np.where(veg_in_poly, vi_data[vi_name], np.nan)
            _update_acc(acc_masked[aoi_id][vi_name], data_masked)


def finalize_records(rows: dict, rows_raw: dict, acc_raw: dict, acc_masked: dict, vi_names: list) -> tuple:
    """Compute final per-AOI VI statistics and return the (masked, raw) record lists."""
    records_vi_masked, records_vi_raw = [], []
    for aoi_id in sorted(rows.keys()):
        row = rows[aoi_id]
        row_raw = rows_raw[aoi_id]
        for vi_name in vi_names:
            vi_key = vi_name.lower()
            row[f"{vi_key}_mean"], row[f"{vi_key}_std"], row[f"{vi_key}_min"], row[f"{vi_key}_max"] = _acc_stats(acc_masked[aoi_id][vi_name])
            row_raw[f"{vi_key}_mean"], row_raw[f"{vi_key}_std"], row_raw[f"{vi_key}_min"], row_raw[f"{vi_key}_max"] = _acc_stats(acc_raw[aoi_id][vi_name])
        records_vi_masked.append(row)
        records_vi_raw.append(row_raw)
    return records_vi_masked, records_vi_raw


def save_results(records_vi_masked: list, records_vi_raw: list, out_dir: str, exp: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame(records_vi_masked).sort_values("aoi_id").reset_index(drop=True)
    out_csv = os.path.join(out_dir, f"{exp}_orthorectified_no_soil.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved VI stats -> {out_csv}")

    df_raw = pd.DataFrame(records_vi_raw).sort_values("aoi_id").reset_index(drop=True)
    out_csv_raw = os.path.join(out_dir, f"{exp}_orthorectified_raw.csv")
    df_raw.to_csv(out_csv_raw, index=False)
    print(f"Saved unmasked VI stats -> {out_csv_raw}")
    print("Done.")


if __name__ == "__main__":
    field = "PPAC-B3"
    parent_folder = f"Data/{field}"
    if field == "Wallpe":
        exps = ["080625_Wallpe", "081325_Wallpe", "081525_Wallpe", "082525_Wallpe", "083025_Wallpe", "091025_Wallpe", "091425_Wallpe"]
    else:
        exps = ["061724_PPAC-B3", "071124_PPAC-B3", "072324_PPAC-B3", "080324_PPAC-B3", "081324_PPAC-B3", "081924_PPAC-B3", "090124_PPAC-B3", "090824_PPAC-B3"]
    aoi_file = f"{field}_aoi_square.geojson"

    # Band order for MicaSense RedEdge-M: B1=Blue, B2=Green, B3=Red, B4=RedEdge, B5=NIR
    vi_names = ["NDVI", "GNDVI", "SAVI", "NDRE", "TVI", "ExG", "SR", "PSRI", "G", "RDVI", "MCARI2", "GRVI", "Red", "Green", "Blue", "RedEdge", "NIR", "OSAVI"]

    # Soil/background removal thresholds (applied using reflectance-normalized bands).
    # Pixels are removed (set to NaN) if HSV 'value' < hsv_v_threshold OR RDVI < rdvi_threshold.
    hsv_v_threshold = 0.05   # HSV value = max(R, G, B) in 0–1 reflectance space
    rdvi_threshold  = 0.15 if field == "PPAC-B3" else 0.10  # RDVI threshold

    utm_crs = "EPSG:32616"
    wgs84_crs = "EPSG:4326"
    utm2wgs_transformer = Transformer.from_crs(utm_crs, wgs84_crs, always_xy=True)

    # Load AOI square polygons, indexed by id
    aoi_square_gdf = gpd.read_file(os.path.join(parent_folder, aoi_file)).set_index("id")

    for exp in exps:
        orthorectified_folder = ensure_orthophotos_exported(parent_folder, exp)
        if orthorectified_folder is None:
            continue

        rows, rows_raw, acc_raw, acc_masked = init_aoi_accumulators(aoi_square_gdf, vi_names, utm2wgs_transformer)

        # Scan through all orthorectified images, find the AOI squares each one intersects, and aggregate stats.
        for orth_path in tqdm(glob.glob(os.path.join(orthorectified_folder, "*.tif")), desc=f"Processing orthophotos ({exp})", unit="photo"):
            process_image(orth_path, aoi_square_gdf, vi_names, hsv_v_threshold, rdvi_threshold,
                           rows, rows_raw, acc_raw, acc_masked)

        records_vi_masked, records_vi_raw = finalize_records(rows, rows_raw, acc_raw, acc_masked, vi_names)
        save_results(records_vi_masked, records_vi_raw, f"./features/{field}", exp)
