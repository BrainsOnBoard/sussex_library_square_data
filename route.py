import numpy as np

from os import path
from rdp import rdp

def load_route(route_name, image_input="unwrapped"):
    # Read decimate script
    decimate_filename = path.join("routes", route_name, "decimate.sh")
    with open(decimate_filename, "r") as decimate_file:
        rdp_epsilon_string = decimate_file.read().replace("\n","")
        rdp_epsilon = float(rdp_epsilon_string.split("=")[1])

    # Read CSV route
    route_data_filename = path.join("routes", route_name, image_input, "database_entries.csv")
    route_data = np.loadtxt(route_data_filename, delimiter=",", skiprows=1, usecols=(0, 1),
                            dtype={"names": ("x", "y"),
                                "formats": (np.float, np.float)})

    # Convert route data from mm to cm
    route_data["x"] /= 10.0
    route_data["y"] /= 10.0

    # Convert coordinates into two column array
    coords = np.vstack((route_data["x"], route_data["y"]))
    coords = np.transpose(coords)

    # Use Ramer-Douglas-Peucker algorithm to decimate path
    remaining_coord_mask = rdp(coords, epsilon=rdp_epsilon, return_mask=True)

    # Extract remaining coords using mask
    remaining_coords = coords[remaining_coord_mask,:]

    # Get indices of remaining points
    remaining_coord_mask_indices = np.where(remaining_coord_mask == True)[0]
    assert remaining_coord_mask_indices[0] == 0
    assert remaining_coord_mask_indices[-1] == (coords.shape[0] - 1)

    return coords, remaining_coords
