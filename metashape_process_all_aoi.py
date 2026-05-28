import glob
import Metashape, os, time
import pandas as pd
import numpy as np
from pyproj import Transformer
from aoi_filtering import get_joined_gdf, calculate_characteristics
from metashape import metashape_pipeline, get_capture_gdf


if __name__ == "__main__":
    field = "PPAC-B3"
    parent_folder = f"Data/{field}"
    aoi_size = 36 if "Wallpe" in parent_folder else 12
    ratio = 0.05

    print(f"Running in {field} field. Using AOI size: {aoi_size} meters")
    # exp = "081525_Wallpe" 
    # for exp in ["061724_PPAC-B3"]:
    for exp in ["061724_PPAC-B3", "071124_PPAC-B3", "072324_PPAC-B3", "080324_PPAC-B3", "081324_PPAC-B3", "081924_PPAC-B3", "090124_PPAC-B3", "090824_PPAC-B3"]:
        aoi_file = "wallpe_aoi.csv" if "Wallpe" in exp else "PPAC-B3_aoi.csv"
        root_folder = os.path.join(parent_folder, exp)
        result_folder = os.path.join(root_folder, "Metashape", "AOI_results")
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)


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

        recorded_info = {"id": [], "num_captures": [], "process_time": []}
        aoi_range = aoi_df['Point_ID'].unique()
        
        for aoi_id in aoi_range:
            try:
                joined_gdf = get_joined_gdf(aoi_df, capture_gdf, width_m, height_m, transformer, target_crs, ratio=ratio, aoi_id=aoi_id, aoi_size=aoi_size)
                print(f"There are {len(joined_gdf)} captures in the AOI {aoi_id} with ratio {ratio}.")

                doc = Metashape.Document()
                try:
                    doc.open(os.path.join(result_folder, f"{exp}_aoi.psx"))
                except:
                    doc.save(os.path.join(result_folder, f"{exp}_aoi.psx"))
                images = [os.path.join(root_folder, "Images", f"{i}_{j}.tif") for i in joined_gdf['image_name'].tolist() for j in range(1, 6)]
                panels = glob.glob(os.path.join(root_folder, "Panel", f"*.tif"))

                start_time = time.time()
                metashape_pipeline(result_folder, doc, images, panels, target_crs, chunk_label=f"AOI_{aoi_id}_ratio_{ratio:.2f}", export=False)
                process_time = time.time() - start_time
                recorded_info["id"].append(aoi_id)
                recorded_info["num_captures"].append(len(joined_gdf))
                recorded_info["process_time"].append(process_time)
            except Exception as e:
                print(f"Error processing AOI {aoi_id}: {e}")
                continue

        results_df = pd.DataFrame(recorded_info)
        results_df.to_csv(os.path.join(result_folder, f"processing_results_each_aoi.csv"), index=False)


