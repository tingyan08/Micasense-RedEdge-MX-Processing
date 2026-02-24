import argparse
import os
import time, json
from pathlib import Path

import numpy as np
import pandas as pd
from mapboxgl.utils import df_to_geojson
from tqdm import tqdm

import micasense.capture as capture
import micasense.imageset as imageset
import cv2

# Camera models: RedEdge-MX



if __name__ == "__main__":

    root_path = Path("091425_Wallpe")

    pan_sharpen = False
    use_dls = True # Downwelling Light Sensor (DLS)
    overwrite = True  # can be set to False to continue interrupted processing
    generateThumbnails = True # set to False to skip generating RGB thumbnails, which can be time consuming for large datasets

    image_path = root_path / 'Images'
    panel_path = root_path / 'Panel'
    outputPath = root_path / 'Processed'


    panelNames = list(panel_path.glob('IMG_0000_*.tif'))
    panelNames = [x.as_posix() for x in panelNames]

    panelCap = capture.Capture.from_filelist(panelNames)

    # destinations on your computer to put the stacks and RGB thumbnails
    outputPath = outputPath.resolve().as_posix()
    print(outputPath)
    thumbnailPath = Path(outputPath) / 'thumbnails'
    thumbnailPath = thumbnailPath.resolve().as_posix()
    print(thumbnailPath)
    capturePath = Path(outputPath) / 'capture'
    capturePath = capturePath.resolve().as_posix()
    print(capturePath)

    cam_model = panelCap.camera_model
    cam_serial = panelCap.camera_serial

    print("Camera model:", cam_model) 

    # if this is a multicamera system like the RedEdge-MX Dual,
    # we can combine the two serial numbers to help identify 
    # this camera system later. 
    if len(panelCap.camera_serials) > 1:
        cam_serial = "_".join(panelCap.camera_serials)
        print("Serial number:", cam_serial)
    else:
        cam_serial = panelCap.camera_serial
        print("Serial number:", cam_serial)



    # Allow this code to align both radiance and reflectance images; but excluding
    # a definition for panelNames above, radiance images will be used
    # For panel images, efforts will be made to automatically extract the panel information
    # but if the panel/firmware is before Altum 1.3.5, RedEdge 5.1.7 the panel reflectance
    # will need to be set in the panel_reflectance_by_band variable.
    # Note: radiance images will not be used to properly create NDVI/NDRE images below.
    if panelCap is not None:
        if panelCap.panel_albedo() is not None and not any(v is None for v in panelCap.panel_albedo()):
            panel_reflectance_by_band = panelCap.panel_albedo()
        else:
            panel_reflectance_by_band = [0.49] * len(panelCap.eo_band_names())  # RedEdge band_index order

        panel_irradiance = panelCap.panel_irradiance(panel_reflectance_by_band)
        img_type = "reflectance"
    else:
        if use_dls:
            img_type = 'reflectance'
        else:
            img_type = "radiance"

    imgset = imageset.ImageSet.from_directory(image_path)

    data, columns = imgset.as_nested_lists()
    df = pd.DataFrame.from_records(data, index='timestamp', columns=columns)

    geojson_data = df_to_geojson(df, columns[3:], lat='latitude', lon='longitude')

    warp_matrices = None
    warp_matrices_filename = root_path / (cam_serial + "_warp_matrices_opencv.npy")

    if warp_matrices_filename.is_file():
        print("Found existing warp matrices for camera", cam_serial)
        load_warp_matrices = np.load(warp_matrices_filename, allow_pickle=True)
        loaded_warp_matrices = []
        for matrix in load_warp_matrices:
            loaded_warp_matrices.append(matrix.astype('float32'))

        warp_matrices = loaded_warp_matrices
        print("Loaded warp matrices from", warp_matrices_filename.resolve())
    else:
        print("No warp matrices found at expected location:", warp_matrices_filename)
        if len(imgset.captures) > 0:
            print("Generating new warp matrices...")
            
            # Select the middle capture for alignment. This is a heuristic - for best results,
            # ensure the middle of your dataset has good features for alignment.
            alignment_capture = imgset.captures[len(imgset.captures)//2]
            
            # Standard OpenCV ECC alignment
            print("Running ECC alignment (this may take several minutes)...")
            
            # Alignment settings from the notebooks
            reference_band = 2  # use the Red band as the reference
            warp_mode = cv2.MOTION_HOMOGRAPHY
            warp_matrices = alignment_capture.get_warp_matrices(ref_index=reference_band)
            
            np.save(warp_matrices_filename, warp_matrices, allow_pickle=True)
            print(f"Saved new warp matrices to: {warp_matrices_filename}")

        else:
            print("ImageSet is empty, cannot generate warp matrices.")

    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    if generateThumbnails and not os.path.exists(thumbnailPath):
        os.makedirs(thumbnailPath)
    if not os.path.exists(capturePath):
        os.makedirs(capturePath)

    # Save out geojson data, so we can open the image capture locations in our GIS
    with open(os.path.join(outputPath, 'imageSet.json'), 'w') as f:
        json.dump(geojson_data, f, indent=2)

    try:
        irradiance = panel_irradiance + [0]
    except NameError:
        irradiance = None

    for i, cap in tqdm(enumerate(imgset.captures), total=len(imgset.captures), desc="Processing captures"):
        outputFilename = cap.images[0].img_name + '.tif'
        thumbnailFilename = cap.images[0].img_name + '.jpg'
        fullOutputPath = os.path.join(capturePath, outputFilename)
        fullThumbnailPath = os.path.join(thumbnailPath, thumbnailFilename)
        if (not os.path.exists(fullOutputPath)) or overwrite:
            if (len(cap.images) == len(imgset.captures[0].images)):
                cap.create_aligned_capture(irradiance_list=irradiance, warp_matrices=warp_matrices)
                cap.save_capture_as_stack(fullOutputPath, pansharpen=pan_sharpen, sort_by_wavelength=False)
                if generateThumbnails:
                    cap.save_capture_as_rgb(fullThumbnailPath)
        cap.clear_image_data()
