import cv2
import sys
import numpy as np
from glob import glob
from os import path, mkdir

def create_sky_mask(mask, raw): 
    ground = (mask == 255) 
    sky_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    sky_mask[ground] = raw[ground] 
    return sky_mask


if len(sys.argv) < 2:
    print("Directory of route or image grid containing unwrapped images and masks expected")
    sys.exit(2)

# Build path to unwrapped files, exit if it doesn't exist
unwrapped_dir = path.join(sys.argv[1], "unwrapped")
if not path.exists(unwrapped_dir):
    print("%s does not exist" % unwrapped_dir)
    sys.exit(1)

# Build path to mask files, exit if it doesn't exist
mask_dir = path.join(sys.argv[1], "mask")
if not path.exists(mask_dir):
    print("%s does not exist" % mask_dir)
    sys.exit(1)

# Build path to skymask files, create if it doesn't exist
skymask_dir = path.join(sys.argv[1], "skymask")
if not path.exists(skymask_dir):
    mkdir(skymask_dir)

# Loop through images
for raw_path in glob(path.join(unwrapped_dir,"unwrapped_*.jpg")):
    print(raw_path)
    
    # Split mask path into directory and filename
    raw_filename = path.split(raw_path)[1]
    
    # Extract title from filename
    raw_title = path.splitext(raw_filename)[0]

    # Build path to mask image
    mask_path = path.join(mask_dir, raw_title + "_mask.png")
    
    # Load corresponding raw and mask image
    raw_image = cv2.imread(raw_path)
    mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    assert raw_image is not None
    assert mask_image is not None
    
    sky_mask = create_sky_mask(mask_image, raw_image)
    sky_mask_path = path.join(skymask_dir, raw_title + "_skymask.png")
    cv2.imwrite(sky_mask_path, sky_mask)
