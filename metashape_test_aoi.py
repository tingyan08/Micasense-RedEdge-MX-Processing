import glob
import Metashape, os, time
import pandas as pd
import numpy as np
from pyproj import Transformer
from aoi_filtering import get_joined_gdf, calculate_characteristics
from metashape import metashape_pipeline, get_capture_gdf


if __name__ == "__main__":
    parent_folder = "Data"
    exp = "090523_PPAC_B3"
    aoi_file = "PPAC_B3_aoi.csv" # PPAC_B3_aoi.csv or wallpe_aoi.csv
    root_folder = os.path.join(parent_folder, exp)
    aoi_id = 121

    gsd_cm, H, W = calculate_characteristics(root_folder)
    width_m = gsd_cm / 100 * W
    height_m = gsd_cm / 100 * H 
    print(f"Calculated GSD: {gsd_cm:.2f} cm/pixel")
    print(f"Image dimensions: {H} x {W} pixels")

    original_crs = "EPSG:4326"  # WGS 84
    target_crs = "EPSG:32616"  # WGS 84 - UTM zone 16N
    transformer = Transformer.from_crs(original_crs, target_crs, always_xy=True)

    aoi_df = pd.read_csv(os.path.join(parent_folder, aoi_file))
    capture_gdf = get_capture_gdf(root_folder)
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
        panels = glob.glob(os.path.join(root_folder, "Panel", f"*.tif"))

        start_time = time.time()
        metashape_pipeline(root_folder, doc, images, panels, target_crs, chunk_label=f"AOI_{aoi_id}_ratio_{ratio:.2f}")
        process_time = time.time() - start_time
        recorded_info["ratio"].append(ratio)
        recorded_info["num_captures"].append(len(joined_gdf))
        recorded_info["process_time"].append(process_time)

    results_df = pd.DataFrame(recorded_info)
    results_df.to_csv(os.path.join(root_folder, "Metashape", f"processing_results_{aoi_id}.csv"), index=False)
