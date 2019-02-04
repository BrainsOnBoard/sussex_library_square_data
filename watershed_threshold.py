import cv2
import sys
import numpy as np
from glob import glob
from os import path, mkdir

# Simple tool for drawing on windows, taken from OpenCV python samples common.py
class Sketcher:
    def __init__(self, windowname, dests, colors_func):
        self.prev_pt = None
        self.windowname = windowname
        self.dests = dests
        self.colors_func = colors_func
        self.dirty = False
        self.show()
        cv2.setMouseCallback(self.windowname, self.on_mouse)

    def show(self):
        cv2.imshow(self.windowname, self.dests[0])

    def on_mouse(self, event, x, y, flags, param):
        pt = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.prev_pt = pt
        elif event == cv2.EVENT_LBUTTONUP:
            self.prev_pt = None

        if self.prev_pt and flags & cv2.EVENT_FLAG_LBUTTON:
            for dst, color in zip(self.dests, self.colors_func()):
                cv2.line(dst, self.prev_pt, pt, color, 3)
            self.dirty = True
            self.prev_pt = pt
            self.show()


cur_marker = 1
colours = np.int32(list(np.ndindex(2, 2, 2))) * 255
def get_colours():
    return list(map(int, colours[cur_marker])), cur_marker

if len(sys.argv) < 2:
    print("Directory of route or image grid containing unwrapped images expected")
    sys.exit(2)

# Build path to unwrapped files, exit if it doesn't exist
unwrapped_dir = path.join(sys.argv[1], "unwrapped")
if not path.exists(unwrapped_dir):
    print("%s does not exist" % unwrapped_dir)
    sys.exit(1)

# Build path to mask files, create if it doesn't exist
mask_dir = path.join(sys.argv[1], "mask")
if not path.exists(mask_dir):
    mkdir(mask_dir)

# Get list of unwrapped images
unwrapped_paths = sorted(glob(path.join(unwrapped_dir,"unwrapped_*.jpg")))

# Loop through unwrapped images
marker_mask_image = None
i = 0;
while i < len(unwrapped_paths):
    print(unwrapped_paths[i])

    # Read input image
    input_image = cv2.imread(unwrapped_paths[i])
    input_draw_image = input_image.copy()
    watershed_mask = None

    # Create matching mask if required
    if marker_mask_image is None:
        marker_mask_image = np.zeros(input_image.shape[:2], dtype=np.int32)
    # Otherwise
    else:
        # Find masked points
        masked_points = marker_mask_image != 0

        # Copy colourised mask onto input draw image
        input_draw_image[masked_points,:] = colours[marker_mask_image[masked_points]]

    # Create sketch
    sketch = Sketcher("Original", [input_draw_image, marker_mask_image], get_colours)
    sketch.dirty = True
    sketch.show()

    while True:
        if sketch.dirty:
            watershed_mask = marker_mask_image.copy()
            cv2.watershed(input_image, watershed_mask)
            overlay = colours[np.maximum(watershed_mask, 0)]
            vis = cv2.addWeighted(input_image, 0.5, overlay, 0.5, 0.0, dtype=cv2.CV_8UC3)
            cv2.imshow("Watershed", vis)
            sketch.dirty = False

        # Process events
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            marker_mask_image[:] = 0
            input_draw_image[:] = input_image
            sketch.dirty = True
            sketch.show()
        elif key >= ord("1") and key <= ord("7"):
            cur_marker = key - ord("0")
        elif key == ord("p") and i > 0:
            i-=1
            break
        elif key == ord("n"):
            # Split mask path into directory and filename
            unwrapped_filename = path.split(unwrapped_paths[i])[1]

            # Extract title from filename
            raw_title = path.splitext(unwrapped_filename)[0]

            # Build path to mask image
            mask_path = path.join(mask_dir, raw_title + "_mask.png")

            min_mask = np.amin(watershed_mask)
            max_mask = np.amax(watershed_mask)

            cv2.imwrite(mask_path, (watershed_mask.astype(np.uint8) - min_mask) * 255 / (max_mask - min_mask))

            i+=1
            break
