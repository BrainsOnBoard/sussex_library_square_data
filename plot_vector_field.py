import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from os import path
from sys import argv

from route import load_route
import plot_settings

# Check we only get a single argument
assert len(argv) == 2

# Read output
output_data = np.loadtxt(argv[1], delimiter=",", skiprows=1, usecols=(0, 1, 2),
                         converters={0: lambda s: float(s[:-3]),
                                     1: lambda s: float(s[:-3]),
                                     2: lambda s: float(s[:-4])},
                         dtype={"names": ("x", "y", "best_heading"),
                                "formats": (np.float, np.float, np.float)})

output_dir, output_filename = path.split(argv[1])
output_title = path.splitext(output_filename)[0]

# Split output file title into pre-defined components
_, route_name, memory, image_input = output_title.split("_")

# Load route
coords, remaining_coords = load_route(route_name, image_input)

# Configure palette
colours = sns.color_palette(n_colors=3)

# Create single-column figure
fig, axis = plt.subplots(figsize=(plot_settings.column_width, 4.0))

# Set axis range to match that of grid
min_x = np.amin(output_data["x"])
max_x = np.amax(output_data["x"])
min_y = np.amin(output_data["y"])
max_y = np.amax(output_data["y"])
axis.set_xlim((min_x, max_x))
axis.set_ylim((min_y, max_y))
axis.set_xticks(np.arange(min_x, max_x, 120))
axis.set_yticks(np.arange(min_y, max_y, 120))

axis.set_xlabel("X [cm]")
axis.set_ylabel("Y [cm]")

plot_settings.remove_axis_junk(axis)

# Plot quivers showing vector field
heading_radians = np.radians(output_data["best_heading"])
u = np.cos(heading_radians)
v = np.sin(heading_radians)
axis.quiver(output_data["x"], output_data["y"], u, v, angles="xy", zorder=3)

# Plot route data
axis.plot(coords[:,0], coords[:,1], zorder=1, color=colours[1])
axis.plot(remaining_coords[:,0], remaining_coords[:,1], zorder=2, color=colours[0])

first_x = remaining_coords[0, 0]
first_y = remaining_coords[0, 1]
dir_x = remaining_coords[1, 0] - first_x
dir_y = remaining_coords[1, 1] - first_y
scale = 100.0 / np.sqrt((dir_x * dir_x) + (dir_y * dir_y))
dir_x *= scale
dir_y *= scale
axis.arrow(first_x - dir_x, first_y - dir_y, dir_x, dir_y,
           color=colours[2], length_includes_head=True, head_width=30.0, zorder=4)




fig.tight_layout()
plt.show()
