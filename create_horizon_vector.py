import cv2
import sys
import numpy as np
from glob import glob
from os import path, mkdir

def create_horizon(mask):
    # Create single row image to hold horizon and initialise to zero
    # **NOTE** this is assuming that missing horizon is basically at the top
    horizon = np.zeros((1, mask.shape[1]), dtype=np.uint8)

    # Loop through columns of mask
    for c in range(mask.shape[1]):
        # Extract column
        mask_column = mask[:,c]

        # Find where horizon i.e. black pixels are
        horizon_coords = np.where(mask_column == 0)[0]

        # If there is a horizon
        if len(horizon_coords) > 0:
            # Take mean to handle noisy horizons and store in horizon
            horizon_average = int(round(np.average(horizon_coords)))
            horizon[0, c] = horizon_average

    return horizon


if len(sys.argv) < 2:
    print("Directory of route or image grid containing unwrapped images and masks expected")
    sys.exit(2)

# Build path to mask files, exit if it doesn't exist
mask_dir = path.join(sys.argv[1], "mask")
if not path.exists(mask_dir):
    print("%s does not exist" % mask_dir)
    sys.exit(1)

# Build path to horizon files, create if it doesn't exist
horizon_dir = path.join(sys.argv[1], "horizon")
if not path.exists(horizon_dir):
    mkdir(horizon_dir)

# Loop through images
for mask_path in glob(path.join(mask_dir,"unwrapped_*_mask.png")):
    print(mask_path)
    
    # Split mask path into directory and filename
    mask_filename = path.split(mask_path)[1]
    
    # Extract title from filename
    mask_title = path.splitext(mask_filename)[0]
    
    # Load corresponding raw and mask image
    mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    assert mask_image is not None
    
    horizon = create_horizon(mask_image)
    horizon_path = path.join(horizon_dir, mask_title[:-5] + "_horizon.png")
    cv2.imwrite(horizon_path, horizon)
