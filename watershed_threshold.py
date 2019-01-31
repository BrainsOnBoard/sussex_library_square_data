import cv2
import numpy as np
from glob import glob

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
                cv2.line(dst, self.prev_pt, pt, color, 5)
            self.dirty = True
            self.prev_pt = pt
            self.show()


cur_marker = 1
colours = np.int32(list(np.ndindex(2, 2, 2))) * 255
def get_colours():
    return list(map(int, colours[cur_marker])), cur_marker

marker_mask_image = None
for f in glob("unwrapped_image_grid/mid_day/*.jpg"):
    # Read input image
    input_image = cv2.imread(f)
    input_draw_image = input_image.copy()
    
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
            m = marker_mask_image.copy()
            cv2.watershed(input_image, m)
            overlay = colours[np.maximum(m, 0)]
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
        elif key == ord("n"):
            break
