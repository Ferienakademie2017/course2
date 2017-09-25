import matplotlib.pyplot as plt
import numpy as np
import os

from utils import get_parameter

VERBOSE = False

def plot_2d_flow_data(filepath, verbose=VERBOSE):
    """Visualize 2D flow field data. If the given input has dimensions (x, y,
    2), the flow velocity is visualized using an arrow plot; if it has
    dimensions (x, y), the flow density is visualized.
    filepath: absolute or relative path to binary NumPy file containing array
    """
    if not os.path.isabs(filepath):
        filepath = os.path.join(os.getcwd(), filepath)

    # parse y-position of obstacle from data filename
    filename, _ = os.path.splitext(os.path.basename(filepath))

    downscaling_factors = get_parameter("downscaling_factors")
    try:
        y_position = float(filename)
        x_position = 2 * get_parameter("relative_x_position") *\
            get_parameter("resolution")
        radius = get_parameter("resolution") *\
            get_parameter("obstacle_radius_factor")
        if "1608" in filepath:
            y_position *= downscaling_factors[1]
            x_position *= downscaling_factors[0]
            radius *= downscaling_factors[0]
    except ValueError as e:
        # if the filename can't be converted to float, set circle parameters
        # to zero to not display it at all
        y_position = x_position = radius = 0
    obstacle = plt.Circle((x_position, y_position), radius)

    data = np.load(filepath)
    if verbose:
        print(data.shape)

    if data.ndim == 3:
        # plot velocity field
        data_x = data[:, :, 0]
        data_y = data[:, :, 1]
        if verbose:
            print(data_x.shape[0])
            print(data_x.shape[1])
            print(data_y.shape[0])
            print(data_y.shape[1])

        x_axis = np.arange(data.shape[1])
        y_axis = np.arange(data.shape[0])

        fig, ax = plt.subplots()
        q = ax.quiver(x_axis, y_axis, data_x, data_y)
    elif data.ndim == 2:
        # plot density
        fig, ax = plt.subplots()
        image = ax.imshow(data, origin="lower")
        fig.colorbar(image)
    else:
        print(data.ndim)

    ax.add_artist(obstacle)
    ax.axis("equal")
    plt.show()

if __name__ == "__main__":
    import sys
    plot_2d_flow_data(sys.argv[1])
