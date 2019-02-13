import cv2
import sys
import numpy as np
from glob import glob
from os import path, mkdir

if len(sys.argv) < 2:
    print("Directory of route or image grid containing masks expected")
    sys.exit(2)

# Build path to mask files, exit if it doesn't exist
mask_dir = path.join(sys.argv[1], "mask")
if not path.exists(mask_dir):
    print("%s does not exist" % mask_dir)
    sys.exit(1)
# Loop through images
for mask_path in glob(path.join(mask_dir,"*_mask.png")):
    print(mask_path)

    # Load corresponding raw and mask image
    mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    assert mask_image is not None

    # Copy second from top and bottom rows to top and bottom
    mask_image[0,1:-1] = mask_image[1,1:-1]
    mask_image[-1,1:-1] = mask_image[-2,1:-1]

    mask_image[:,0] = mask_image[:,1]
    mask_image[:,-1] = mask_image[:,-2]

    # Overwrite image
    cv2.imwrite(mask_path, mask_image)
