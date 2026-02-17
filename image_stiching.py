# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import sys, os, glob


def main(image_path):
    # Read the images to be stitched
    imgs = glob.glob(os.path.join(image_path, "*.jpg"))
    # read input images
    imgs = []
    for img_name in args.img:
        img = cv.imread(cv.samples.findFile(img_name))
        if img is None:
            print("can't read image " + img_name)
            sys.exit(-1)
        imgs.append(img)

    #![stitching]
    stitcher = cv.Stitcher.create(args.mode)
    status, pano = stitcher.stitch(imgs)

    if status != cv.Stitcher_OK:
        print("Can't stitch images, error code = %d" % status)
        sys.exit(-1)
    #![stitching]

    cv.imwrite(args.output, pano)
    print("stitching completed successfully. %s saved!" % args.output)

    print('Done')


if __name__ == '__main__':
    root_folder = "B1"
    geojson = os.path.join(root_folder, "B1.geojson")
    
    main(image_path)
    cv.destroyAllWindows()