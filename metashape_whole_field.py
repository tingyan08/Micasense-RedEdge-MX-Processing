import glob
import traceback
import Metashape, os, time
from pyproj import Transformer
from metashape import metashape_pipeline, get_capture_gdf



if __name__ == "__main__":
    parent_folder = "Data/PPAC-B3/2021"
    for root_folder in sorted(glob.glob(os.path.join(parent_folder, "071421_PPAC-B3"))):
        exp = os.path.basename(root_folder)
        try:
            result_folder = os.path.join(root_folder, "Metashape", "whole_field")
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)

            original_crs = "EPSG:4326"  # WGS 84
            target_crs = "EPSG:32616"  # WGS 84 - UTM zone 16N
            transformer = Transformer.from_crs(original_crs, target_crs, always_xy=True)

            capture_gdf = get_capture_gdf(root_folder)
            capture_gdf = capture_gdf.to_crs(target_crs)

            doc = Metashape.Document()
            try:
                doc.open(os.path.join(result_folder, f"{exp}_whole_field.psx"))
            except:
                doc.save(os.path.join(result_folder, f"{exp}_whole_field.psx"))
            images = [os.path.join(root_folder, "Images", f"{i}_{j}.tif") for i in capture_gdf['image_name'].tolist() for j in range(1, 6)]
            panels = glob.glob(os.path.join(root_folder, "Panel", f"*.tif"))

            start_time = time.time()
            metashape_pipeline(result_folder, doc, images, panels, target_crs, chunk_label=f"full_field", export=False)
            process_time = time.time() - start_time
            print(f"Processing time for the whole field: {process_time:.2f} seconds")
        except Exception:
            print(f"[ERROR] Failed to process {exp}:")
            traceback.print_exc()
            continue
