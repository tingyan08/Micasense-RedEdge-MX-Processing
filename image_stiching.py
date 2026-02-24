# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
import sys, os, glob
import geopandas as gpd
from time import time


def main(root_folder, imgs):
    imgs = [cv2.imread(img) for img in imgs]

    #![stitching]
    stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
    status, pano = stitcher.stitch(imgs)

    if status != cv2.Stitcher_OK:
        print("Can't stitch images, error code = %d" % status)
        sys.exit(-1)
    #![stitching]

    output_file = os.path.join(root_folder, "B1_stitched.jpg")
    cv2.imwrite(output_file, pano)
    print("stitching completed successfully. %s saved!" % output_file)

    print('Done')


if __name__ == '__main__':
    root_folder = "B1"
    geojson = os.path.join(root_folder, "Processed", "imageSet.json")
    gdf = gpd.read_file(geojson)
    gdf['img_path'] = gdf['capture_id'].apply(lambda x: os.path.normpath(os.path.join(root_folder, "Processed", "thumbnails", f"{x}.jpg")))
    img_list = gdf['img_path'].tolist()
    print(f"Found {len(img_list)} images for stitching.")
    start_time = time()
    main(root_folder, img_list[:100])  # Process only the first 10 images for testing
    end_time = time()
    print(f"Stitching took {end_time - start_time:.2f} seconds.")
    cv2.destroyAllWindows()