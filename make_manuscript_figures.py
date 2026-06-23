import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from shapely.geometry import box
from pyproj import Transformer

from aoi_filtering import calculate_characteristics

OUT_DIR = "../God_UAV_data_Manuscript/figures"


def make_expanded_window_figure(field="PPAC-B3", exp="081324_PPAC-B3", aoi_id=51,
                                ratio=0.15, out_name="expanded_window.png"):
    """Real-data illustration of expanded-window capture selection for one grid."""
    parent_folder = f"Data/{field}"
    root_folder = os.path.join(parent_folder, exp)
    aoi_size = 12 if field == "PPAC-B3" else 36

    gsd_cm, H, W = calculate_characteristics(root_folder)
    width_m = gsd_cm / 100 * W
    height_m = gsd_cm / 100 * H

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32616", always_xy=True)
    aoi_df = pd.read_csv(os.path.join(parent_folder, f"{field}_aoi.csv"))
    row = aoi_df[aoi_df["Point_ID"] == aoi_id].iloc[0]
    cx, cy = transformer.transform(row["Longitude"], row["Latitude"])

    aoi_poly = box(cx - aoi_size / 2, cy - aoi_size / 2, cx + aoi_size / 2, cy + aoi_size / 2)
    expanded = box(cx - aoi_size / 2 - (0.5 - ratio) * width_m,
                   cy - aoi_size / 2 - (0.5 - ratio) * height_m,
                   cx + aoi_size / 2 + (0.5 - ratio) * width_m,
                   cy + aoi_size / 2 + (0.5 - ratio) * height_m)

    capture_gdf = gpd.read_file(os.path.join(root_folder, "imageSet.geojson")).to_crs("EPSG:32616")
    capture_gdf["x"] = capture_gdf.geometry.x
    capture_gdf["y"] = capture_gdf.geometry.y

    sel = capture_gdf.geometry.within(expanded)
    selected = capture_gdf[sel]

    exminx, exminy, exmaxx, exmaxy = expanded.bounds
    pad = max(width_m, height_m) * 0.7
    view = box(exminx - pad, exminy - pad, exmaxx + pad, exmaxy + pad)
    vminx, vminy, vmaxx, vmaxy = view.bounds
    near = capture_gdf[capture_gdf.geometry.within(view)]

    fig, ax = plt.subplots(figsize=(7.5, 7.5))

    # footprints of selected captures (semi-transparent, show overlap)
    for _, c in selected.iterrows():
        ax.add_patch(Rectangle((c["x"] - width_m / 2, c["y"] - height_m / 2), width_m, height_m,
                               facecolor="tab:green", edgecolor="tab:green", alpha=0.06, lw=0.5))

    # all nearby capture centers
    ax.scatter(near["x"], near["y"], s=14, c="0.6", label="captures (not selected)", zorder=3)
    ax.scatter(selected["x"], selected["y"], s=26, c="tab:green",
               edgecolors="k", linewidths=0.4, label=f"selected captures (n={len(selected)})", zorder=4)

    # expanded window
    ax.add_patch(Rectangle((expanded.bounds[0], expanded.bounds[1]),
                           expanded.bounds[2] - expanded.bounds[0],
                           expanded.bounds[3] - expanded.bounds[1],
                           fill=False, edgecolor="tab:blue", lw=2, ls="--",
                           label=f"expanded window (r={ratio:.2f})", zorder=5))
    # AOI grid square
    ax.add_patch(Rectangle((cx - aoi_size / 2, cy - aoi_size / 2), aoi_size, aoi_size,
                           fill=False, edgecolor="tab:red", lw=2.2,
                           label=f"target grid ({aoi_size} m)", zorder=6))

    ax.set_xlim(vminx, vmaxx)
    ax.set_ylim(vminy, vmaxy)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (m, UTM 16N)")
    ax.set_ylabel("Northing (m, UTM 16N)")
    ax.set_title(f"Expanded-window capture selection (grid {aoi_id}, {exp})")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, out_name)
    plt.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"Saved {path}  (selected {len(selected)} / {len(capture_gdf)} captures)")


def make_three_strategies_schematic(out_name="three_strategies.png"):
    """Conceptual schematic contrasting the three VI-extraction strategies."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.2), layout="constrained")
    grid_c = "tab:red"
    foot_c = "tab:green"

    def draw_grid_cells(ax, x0, y0, n, w, highlight=None):
        for i in range(n):
            for j in range(n):
                hl = highlight is not None and (i, j) == highlight
                ax.add_patch(Rectangle((x0 + i * w, y0 + j * w), w, w, fill=hl,
                                       facecolor="tab:red" if hl else "none", alpha=0.25,
                                       edgecolor="0.5", lw=0.8))

    # ---- Strategy 1: individual orthorectified images ----
    ax = axes[0]
    offsets = [(-0.18, 0.12), (0.0, 0.0), (0.2, -0.1), (0.1, 0.22)]
    for k, (dx, dy) in enumerate(offsets):
        ax.add_patch(Rectangle((0.18 + dx, 0.2 + dy), 0.55, 0.45, fill=True,
                               facecolor=foot_c, alpha=0.10, edgecolor=foot_c, lw=1.2))
    ax.add_patch(Rectangle((0.4, 0.4), 0.16, 0.16, fill=False, edgecolor=grid_c, lw=2.4))
    ax.text(0.48, 0.34, "grid", color=grid_c, ha="center", fontsize=9, weight="bold")
    ax.annotate("", xy=(0.85, 0.42), xytext=(0.62, 0.48),
                arrowprops=dict(arrowstyle="->", color="k"))
    ax.text(0.5, 0.86, "multiple overlapping captures,\nno blending", ha="center", fontsize=9)
    ax.text(0.5, 0.06, "$\\rightarrow$ several VI observations / grid", ha="center", fontsize=9.5,
            color="k")
    ax.set_title("Strategy 1\nIndividual orthorectified images", fontsize=10.5)

    # ---- Strategy 2: full-field orthomosaic ----
    ax = axes[1]
    ax.add_patch(Rectangle((0.08, 0.12), 0.84, 0.7, fill=True, facecolor=foot_c, alpha=0.18,
                           edgecolor=foot_c, lw=1.5))
    draw_grid_cells(ax, 0.12, 0.16, 5, 0.16, highlight=(2, 2))
    ax.text(0.5, 0.9, "entire field stitched into\none orthomosaic", ha="center", fontsize=9)
    ax.text(0.5, 0.06, "$\\rightarrow$ one VI value / grid", ha="center", fontsize=9.5)
    ax.set_title("Strategy 2\nFull-field orthomosaic", fontsize=10.5)

    # ---- Strategy 3: expanded-window local reconstruction ----
    ax = axes[2]
    ax.add_patch(Rectangle((0.3, 0.3), 0.4, 0.4, fill=False, edgecolor="tab:blue", lw=2, ls="--"))
    for dx, dy in [(-0.05, 0.04), (0.06, -0.03), (0.02, 0.08), (-0.02, -0.06)]:
        ax.add_patch(Rectangle((0.34 + dx, 0.34 + dy), 0.32, 0.32, fill=True,
                               facecolor=foot_c, alpha=0.10, edgecolor=foot_c, lw=1.0))
    ax.add_patch(Rectangle((0.43, 0.43), 0.14, 0.14, fill=False, edgecolor=grid_c, lw=2.4))
    ax.text(0.5, 0.74, "expanded window", color="tab:blue", ha="center", fontsize=8.5)
    ax.text(0.5, 0.9, "local mosaic from nearby\ncaptures only", ha="center", fontsize=9)
    ax.text(0.5, 0.06, "$\\rightarrow$ grid clipped from window center", ha="center", fontsize=9.5)
    ax.set_title("Strategy 3\nExpanded-window reconstruction", fontsize=10.5)

    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(ax.get_title(), pad=16)

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, out_name)
    plt.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    make_three_strategies_schematic()
    make_expanded_window_figure()
