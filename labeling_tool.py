import os
from glob import glob
from collections import namedtuple

import cv2
import errno
import pandas
import numpy as np

# ------------------------------ PARAMETERS ------------------------------ #
# True - to do labeling, False - to view labeled data(from output_folder)
labeling_mode = True
# folder with images to be labeled
input_folder = 'unwrapped_image_grid/mid_day'
# folder to store labeled images and .csv file
output_folder = 'labeled_images/mid_day'
# grid size in slabs column-major
grid_size = (13, 16)
# length of slab's sides (x_scale, y_scale) in cm
scale = (40, 40)
# heading of camera in degrees with respect to 0-angle;
heading = -90
# ------------------------------------------------------------------------ #

# -------------------------- INTERNAL PARAMETERS ------------------------- #
# --- sizes
grid_image_border_width = 5
grid_image_cell_size = (60, 60)
grid_image_size = (grid_size[1] * (grid_image_cell_size[1] + grid_image_border_width),
                   grid_size[0] * (grid_image_cell_size[0] + grid_image_border_width))
cell_point_1 = lambda x, y: (x * (grid_image_cell_size[0] + grid_image_border_width) \
                             + grid_image_border_width // 2, \
                             y * (grid_image_cell_size[1] + grid_image_border_width) \
                             + grid_image_border_width // 2)
cell_point_2 = lambda x, y: ((x + 1) * (grid_image_cell_size[0] + grid_image_border_width),
                             (y + 1) * (grid_image_cell_size[1] + grid_image_border_width))
cell_text_point = lambda x, y: (x * (grid_image_cell_size[0] + grid_image_border_width) \
                                + grid_image_border_width // 2, \
                                (y + 1) * (grid_image_cell_size[1] + grid_image_border_width) \
                                - grid_image_border_width)
# --- fonts
cell_font = cv2.FONT_HERSHEY_SIMPLEX
cell_font_scale = .5
# --- colors
empty_cell_color = (0, 0, 255)
filled_cell_color = (0, 255, 0)
cell_font_color = (255, 255, 255)
# --- structures
GridCell = namedtuple('GridCell', ['x', 'y', 'heading', 'filename'])
thread_pause = True
# ------------------------------------------------------------------------ #

def click_handler(event, x0, y0, flags, param):
    global thread_pause
    current_image, grid = param
    if event == cv2.EVENT_MBUTTONDOWN:
        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                if cell_point_1(x, y)[0] < x0 < cell_point_2(x, y)[0] \
                   and cell_point_1(x, y)[1] < y0 < cell_point_2(x, y)[1]:
                    if grid[x, grid_size[1] - y - 1]:
                        grid[x, grid_size[1] - y - 1] = None
                        real_x = scale[0] * x
                        real_y = scale[1] * (grid_size[1] - y - 1)
                        filename = output_folder + '/' + str(real_x) + '_' + str(real_y) + '.jpg'
                        os.remove(filename)
                        show_grid(grid)
                    return
    elif event == cv2.EVENT_LBUTTONDOWN:
        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                if cell_point_1(x, y)[0] < x0 < cell_point_2(x, y)[0] \
                   and cell_point_1(x, y)[1] < y0 < cell_point_2(x, y)[1]:
                    real_x = scale[0] * x
                    real_y = scale[1] * (grid_size[1] - y - 1)
                    filename = str(real_x) + '_' + str(real_y) + '.jpg'
                    if grid[x, grid_size[1] - y - 1]:
                        preview_image = cv2.imread(output_folder + '/' + filename)
                        cv2.imshow('preview | labeling tool', preview_image)
                        show_grid(grid)
                    else:
                        cv2.imwrite(output_folder + '/' + filename, current_image)
                        grid[x, grid_size[1] - y - 1] = GridCell(real_x, real_y, heading, filename)
                        thread_pause = False
                    return


def show_grid(grid):
    grid_image = np.zeros(shape=grid_image_size + (3,), dtype=np.float32)
    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            if grid[x, (grid_size[1] - y - 1)]:
                cell_color = filled_cell_color
            else:
                cell_color = empty_cell_color
            grid_image = cv2.rectangle(grid_image,
                                       cell_point_1(x, y),
                                       cell_point_2(x, y),
                                       cell_color,
                                       -1)
            cell_text = str(x) + ', ' + str(grid_size[1] - y - 1)
            grid_image = cv2.putText(grid_image,
                                     cell_text,
                                     cell_text_point(x, y),
                                     cell_font,
                                     cell_font_scale,
                                     cell_font_color)
    cv2.imshow('grid | labeling tool', grid_image)


def labeling():
    global input_folder, output_folder, thread_pause
    if input_folder.endswith('/'):
        input_folder = input_folder[:-1]
    if output_folder.endswith('/'):
        output_folder = output_folder[:-1]

    try:
        os.makedirs(output_folder)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    data_frame = pandas.DataFrame(columns=['x [cm]', 'y [cm]', 'heading [degs]', 'filename'])

    filenames = glob(input_folder + '/*.jpg')
    image_shape = cv2.imread(filenames[0]).shape
    grid = np.empty(shape=grid_size, dtype=GridCell)
    current_image = np.empty(image_shape, dtype=np.uint8)

    cv2.namedWindow('grid | labeling tool')
    cv2.setMouseCallback('grid | labeling tool', click_handler, [current_image, grid])

    try:
        for filename in filenames:
            current_image[:] = cv2.imread(filename)
            cv2.imshow('current image | labeling tool', current_image)
            show_grid(grid)
            thread_pause = True
            while thread_pause:
                cv2.waitKey(33)
    except KeyboardInterrupt:
        pass

    np.flip(grid, 0)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j]:
                data_frame = data_frame.append({'x [cm]': grid[i, j].x,
                                                'y [cm]': grid[i, j].y,
                                                'heading [degs]': grid[i, j].heading,
                                                'filename': grid[i, j].filename},
                                                ignore_index=True)
    data_frame.to_csv(output_folder + '/grid_data.csv')


def view():
    pass


def main():
    if labeling_mode:
        labeling()
    else:
        view()


if __name__ == '__main__':
    main()
