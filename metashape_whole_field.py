import glob
import Metashape, os, time
from pyproj import Transformer
from metashape import metashape_pipeline, get_capture_gdf



if __name__ == "__main__":
    parent_folder = "Data"
    exp = "090523_PPAC_B3"
    root_folder = os.path.join(parent_folder, exp)

    original_crs = "EPSG:4326"  # WGS 84
    target_crs = "EPSG:32616"  # WGS 84 - UTM zone 16N
    transformer = Transformer.from_crs(original_crs, target_crs, always_xy=True)

    capture_gdf = get_capture_gdf(root_folder)
    capture_gdf = capture_gdf.to_crs(target_crs)

    doc = Metashape.Document()
    try:
        doc.open(os.path.join(root_folder, "Metashape", f"{exp}.psx"))
    except:
        doc.save(os.path.join(root_folder, "Metashape", f"{exp}.psx"))
    images = [os.path.join(root_folder, "Images", f"{i}_{j}.tif") for i in capture_gdf['image_name'].tolist() for j in range(1, 6)]
    panels = glob.glob(os.path.join(root_folder, "Panel", f"*.tif"))

    start_time = time.time()
    metashape_pipeline(root_folder, doc, images, panels, target_crs, chunk_label=f"full_field")
    process_time = time.time() - start_time
    print(f"Processing time for the whole field: {process_time:.2f} seconds")
