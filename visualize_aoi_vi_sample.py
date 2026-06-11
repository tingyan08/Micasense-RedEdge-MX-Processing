import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import box
import matplotlib.pyplot as plt

from extract_raster_orthorectified_from_whole import get_intersecting_aois, get_aoi_pixel_masks


def compute_ndvi(b3, b5):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (b5 - b3) / (b5 + b3)


def visualize_single_aoi(field, exp, aoi_square_gdf, hsv_v_threshold, rdvi_threshold, out_path="aoi_vi_sample.png"):
    """RGB / raw NDVI / vegetation-masked NDVI for one AOI square, using one image that fully contains it."""
    parent_folder = f"Data/{field}"
    orthorectified_folder = os.path.join(parent_folder, exp, "Metashape", "whole_field", "Orthorectified")

    # Find an image that fully contains at least one AOI square
    image_name, aoi_id = None, None
    for candidate in sorted(os.listdir(orthorectified_folder)):
        orth = os.path.join(orthorectified_folder, candidate)
        with rasterio.open(orth) as src:
            bounds = src.bounds
            ortho_crs = src.crs
        aoi_proj = aoi_square_gdf.to_crs(ortho_crs)
        bbox_gdf = gpd.GeoDataFrame(geometry=[box(*bounds)], crs=ortho_crs)
        contained = gpd.sjoin(aoi_proj, bbox_gdf, how="inner", predicate="within")
        if not contained.empty:
            image_name = candidate
            aoi_id = contained.index[0]
            break

    if image_name is None:
        raise RuntimeError("No orthophoto fully contains an AOI square.")

    print(f"[single AOI] Using image {image_name} for AOI {aoi_id}")

    orth = os.path.join(orthorectified_folder, image_name)
    with rasterio.open(orth) as src:
        ortho = src.read()
        transform = src.transform
        ortho_crs = src.crs

    aoi_poly = gpd.GeoSeries([aoi_square_gdf.loc[aoi_id].geometry], crs=aoi_square_gdf.crs).to_crs(ortho_crs).iloc[0]

    b1 = ortho[0].astype(np.float64) / 32768  # Blue
    b2 = ortho[1].astype(np.float64) / 32768  # Green
    b3 = ortho[2].astype(np.float64) / 32768  # Red
    b5 = ortho[4].astype(np.float64) / 32768  # NIR

    nodata_mask = ortho[5] == 0 if ortho.shape[0] >= 6 else np.zeros_like(b1, dtype=bool)

    hsv_v = np.maximum(np.maximum(b3, b2), b1)
    denom = np.sqrt(np.maximum(b5 + b3, 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        rdvi = np.where(denom > 0, (b5 - b3) / denom, np.nan)
    veg_mask = (hsv_v >= hsv_v_threshold) & (rdvi >= rdvi_threshold) & ~nodata_mask

    ndvi = compute_ndvi(b3, b5)
    ndvi[nodata_mask] = np.nan

    poly_mask = geometry_mask([aoi_poly], out_shape=ndvi.shape, transform=transform, invert=True)

    # Crop to the AOI's bounding box (with a small margin) for display
    rows_idx = np.where(np.any(poly_mask, axis=1))[0]
    cols_idx = np.where(np.any(poly_mask, axis=0))[0]
    pad = 5
    rmin, rmax = max(rows_idx[0] - pad, 0), min(rows_idx[-1] + pad, ndvi.shape[0] - 1)
    cmin, cmax = max(cols_idx[0] - pad, 0), min(cols_idx[-1] + pad, ndvi.shape[1] - 1)

    sl = (slice(rmin, rmax + 1), slice(cmin, cmax + 1))

    rgb_crop = np.stack([b3[sl], b2[sl], b1[sl]], axis=-1)
    rgb_crop = np.clip(rgb_crop / np.nanpercentile(rgb_crop, 98), 0, 1)

    ndvi_crop = ndvi[sl]
    poly_mask_crop = poly_mask[sl]
    veg_mask_crop = veg_mask[sl]

    ndvi_raw = np.where(poly_mask_crop, ndvi_crop, np.nan)
    ndvi_masked = np.where(poly_mask_crop & veg_mask_crop, ndvi_crop, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(rgb_crop)
    axes[0].contour(poly_mask_crop, levels=[0.5], colors="red", linewidths=1.5)
    axes[0].set_title(f"RGB crop\nAOI {aoi_id} ({image_name})")

    im1 = axes[1].imshow(ndvi_raw, cmap="RdYlGn", vmin=-1, vmax=1)
    axes[1].set_title("NDVI (all AOI pixels)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(ndvi_masked, cmap="RdYlGn", vmin=-1, vmax=1)
    axes[2].set_title("NDVI (vegetation-masked)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def visualize_image_aoi_localization(field, exp, image_name, aoi_square_gdf, out_path="image_aoi_localization_sample.png"):
    """For one orthorectified image that overlaps several AOI squares, show the RGB image with the
    AOI grid overlaid, and a pixel map colored by which AOI square each pixel is assigned to."""
    parent_folder = f"Data/{field}"
    orthorectified_folder = os.path.join(parent_folder, exp, "Metashape", "whole_field", "Orthorectified")
    orth = os.path.join(orthorectified_folder, image_name)

    with rasterio.open(orth) as src:
        ortho = src.read()
        transform = src.transform
        ortho_crs = src.crs
        bounds = src.bounds

    intersecting = get_intersecting_aois(bounds, ortho_crs, aoi_square_gdf)
    if intersecting.empty:
        raise RuntimeError(f"{image_name} does not intersect any AOI square.")

    out_shape = ortho.shape[1:]
    aoi_pixel_masks = get_aoi_pixel_masks(intersecting, transform, out_shape)
    print(f"[localization] {image_name} overlaps AOI squares: {sorted(aoi_pixel_masks.keys())}")

    b1 = ortho[0].astype(np.float64) / 32768  # Blue
    b2 = ortho[1].astype(np.float64) / 32768  # Green
    b3 = ortho[2].astype(np.float64) / 32768  # Red
    nodata_mask = ortho[5] == 0 if ortho.shape[0] >= 6 else np.zeros_like(b1, dtype=bool)

    rgb = np.stack([b3, b2, b1], axis=-1)
    rgb = np.clip(rgb / np.nanpercentile(rgb[~nodata_mask], 98), 0, 1)

    # Build a label map: each pixel gets the id of the AOI square it falls in (0 = none)
    aoi_ids = sorted(aoi_pixel_masks.keys())
    label_map = np.zeros(out_shape, dtype=np.int32)
    for label, aoi_id in enumerate(aoi_ids, start=1):
        label_map[aoi_pixel_masks[aoi_id]] = label

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    axes[0].imshow(rgb)
    axes[0].set_title(f"RGB orthophoto\n{image_name}")
    cmap_grid = plt.get_cmap("tab20", max(len(aoi_ids), 1))
    for label, aoi_id in enumerate(aoi_ids, start=1):
        color = cmap_grid(label - 1)
        axes[0].contour(aoi_pixel_masks[aoi_id], levels=[0.5], colors=[color], linewidths=1.5)
        ys, xs = np.where(aoi_pixel_masks[aoi_id])
        axes[0].text(xs.mean(), ys.mean(), str(aoi_id), color=color, fontsize=10,
                      ha="center", va="center", weight="bold")

    label_map_masked = np.ma.masked_equal(label_map, 0)
    cmap_label = plt.get_cmap("tab20", max(len(aoi_ids), 1)).copy()
    cmap_label.set_bad(color="white")
    im = axes[1].imshow(label_map_masked, cmap=cmap_label, vmin=0.5, vmax=len(aoi_ids) + 0.5)
    axes[1].set_title("Pixel → AOI grid assignment")
    cbar = plt.colorbar(im, ax=axes[1], ticks=range(1, len(aoi_ids) + 1), fraction=0.046)
    cbar.ax.set_yticklabels([str(aid) for aid in aoi_ids])
    cbar.set_label("AOI id")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    field = "PPAC-B3"
    parent_folder = f"Data/{field}"
    aoi_file = f"{field}_aoi_square.geojson"

    hsv_v_threshold = 0.05
    rdvi_threshold = 0.15

    aoi_square_gdf = gpd.read_file(os.path.join(parent_folder, aoi_file)).set_index("id")

    visualize_single_aoi(field, "061724_PPAC-B3", aoi_square_gdf, hsv_v_threshold, rdvi_threshold)
    visualize_image_aoi_localization(field, "081324_PPAC-B3", "IMG_0100_1.tif", aoi_square_gdf)
