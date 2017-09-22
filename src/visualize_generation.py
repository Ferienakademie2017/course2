import scipy.misc as misc
import numpy as np
from numpy import linalg as LA
import math

import matplotlib.pyplot as plt


def create_and_save_figure(path, name):
    net_flow = np.load(path)
    net_flow = net_flow.transpose((1, 0, 2))
    image_size = net_flow.shape

    skip = 2
    X, Y = np.mgrid[0:image_size[0]:skip, 0:image_size[1]:skip]

    plt.quiver(X, Y, net_flow[::skip, ::skip, 0], net_flow[::skip, ::skip, 1], units='inches')
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')
    plt.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        left='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelleft='off')
    # plt.set_xlim(0, image_size[0])
    # plt.set_ylim(0, image_size[1])

    plt.savefig("../res/visualization/{}.png".format(name))
    plt.cla()
    #plt.show()

if __name__ == "__main__":
    for i in range(2000):
        create_and_save_figure("../res/visualization_data/{}.npy".format(i), i)