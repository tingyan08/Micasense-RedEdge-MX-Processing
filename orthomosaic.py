import os
import micasense.metadata as metadata
from pathlib import Path
from pyodm import Node

def calculate_gsd(folder):
    img_dir = Path(folder) / "Images"
    panel_dir = Path(folder) / "Panel"
    img = next(Path(img_dir).glob("*.tif"))  # Assuming all images have the same altitude
    panel = next(Path(panel_dir).glob("*.tif"))  # Assuming all panels have the same altitude


    # Calculate AGL (Above Ground Level) using the altitude from the image and panel metadata
    img_meta = metadata.Metadata(img.resolve().as_posix())
    panel_meta = metadata.Metadata(panel.resolve().as_posix())
    image_alt = img_meta.position()[2]
    panel_alt = panel_meta.position()[2]
    agl = image_alt - panel_alt # AGL in meters
    agl = agl * 100  # Convert AGL to centimeters for GSD calculation
    print(f"Calculated AGL: {agl:.2f} cm")

    # Get aligned image metadata to confirm altitude after processing
    aligned_file = next((Path(folder) / "Processed" / "capture").glob("*.tif"))
    aligned_meta = metadata.Metadata(aligned_file.resolve().as_posix())
    # Sensor width and focal length 
    pixel_size = 1 / aligned_meta.get_item("EXIF:XResolution")  # in mm/pixel
    print(f"Pixel size: {pixel_size:.4f} mm/pixel")
    focal_length = aligned_meta.get_item("EXIF:FocalLength")  # in mm
    print(f"Focal length: {focal_length:.4f} mm")
    gsd = agl * pixel_size / focal_length  # GSD in mm/pixel

    return gsd

if __name__ == "__main__":
    # Connect to your local Docker node 
    n = Node('localhost', 3000)

    # Point to the folder where your script saved the .tif stacks
    folder = "B1"
    stack_folder = Path(folder) / "Processed" / "capture"
    images = [p.resolve().as_posix() for p in stack_folder.glob("*.tif")]
    print(f"Found {len(images)} aligned images for processing.")

    # Calculate GSD based on AGL
    gsd = calculate_gsd(folder)
    print(f"Calculated GSD: {gsd:.2f} cm/pixel")

    # Create the task with Multispectral-specific options
    options = {
        "orthophoto-resolution": gsd, # Adjust based on your flight height
        "dsm": True,                  # Useful for elevation data
        "multispectral": True,        # Tells ODM to handle multiple bands
        "feature-quality": "high",    # Better for vegetation/fields
        "pc-quality": "medium",       # Saves RAM
    }
    task = n.create_task(images, options)

    print(f"Task created: {task.uuid}. Processing...")
    task.wait_for_completion()

    # Download the final Orthomosaic
    task.download_assets("./final_results")
    print("Done! Check the 'final_results' folder.")