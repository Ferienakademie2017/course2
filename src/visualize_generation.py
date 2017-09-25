import numpy as np
import matplotlib.pyplot as plt

_, ax = plt.subplots(figsize=(64 / 8, 32 / 8))
ax.tick_params(
    axis='x',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    bottom='off',  # ticks along the bottom edge are off
    top='off',  # ticks along the top edge are off
    labelbottom='off')
ax.tick_params(
    axis='y',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    left='off',  # ticks along the bottom edge are off
    top='off',  # ticks along the top edge are off
    labelleft='off')


def create_and_save_figure(path, name):
    net_flow = np.load(path).transpose(1, 0, 2)
    image_size = net_flow.shape

    skip = 2
    X, Y = np.mgrid[0:image_size[0]:skip, 0:image_size[1]:skip]

    ax.quiver(X, Y, net_flow[::skip, ::skip, 0], net_flow[::skip, ::skip, 1], units='inches')

    plt.savefig("../res/visualization/{}.png".format(name))
    ax.cla()
    # plt.show()


if __name__ == "__main__":
    for i in range(2000):
        create_and_save_figure("../res/visualization_data/{}.npy".format(i), i)
        if i % 100 == 0: print("Processed {}%".format(i * 10 / 2000))
