import matplotlib.pyplot as plt
import numpy as np
import os

VERBOSE = False

def plot_2d_velocities(filepath, verbose=VERBOSE):
    """Visualize 2D velocity field. 
    filepath: absolute or relative path to binary NumPy file containing array of
        dimensions (x, y, 2).
    """
    if not os.path.isabs(filepath):
        filepath = os.path.join(os.getcwd(), filepath)

    data = np.load(filepath)
    if verbose:
        print(data.shape[0])
        print(data.shape[1])
        print(data.shape[2])

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

    plt.show()

if __name__ == "__main__":
    import sys
    plot_2d_velocities(sys.argv[1])
