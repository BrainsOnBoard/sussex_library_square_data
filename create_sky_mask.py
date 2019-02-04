import cv2
import numpy as np
from glob import glob
from os import path

def create_sky_mask(mask, raw): 
    ground = (mask == 255) 
    sky_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    sky_mask[ground] = raw[ground] 
    return sky_mask

# Loop through images
for raw_path in glob("unwrapped_image_grid/mid_day/*.jpg"):
    print(raw_path)
    
    # Split mask path into directory and filename
    raw_dir, raw_filename = path.split(raw_path)
    
    # Extract title from filename
    raw_title = path.splitext(raw_filename)[0]

    # Build path to mask image
    mask_path = path.join(raw_dir, raw_title + "_mask.png")
    
    # Load corresponding raw and mask image
    raw_image = cv2.imread(raw_path)
    mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    assert raw_image is not None
    assert mask_image is not None
    
    sky_mask = create_sky_mask(mask_image, raw_image)
    sky_mask_path = path.join(raw_dir, raw_title + "_skymask.png")
    cv2.imwrite(sky_mask_path, sky_mask)
