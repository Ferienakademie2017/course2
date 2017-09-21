import os
import os.path
import math
import scipy
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt

import pickle

def ensureDir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def writeToFile(filename, content):
    ensureDir(filename)
    file = open(filename, 'w')
    file.truncate()
    file.write(content)
    file.close()


def readFromFile(filename):
    if not os.path.isfile(filename):
        return ''

    file = open(filename, 'r')
    result = file.read()
    file.close()
    return result


def serialize(filename, obj):
    ensureDir(filename)
    file = open(filename, 'wb')
    pickle.dump(obj, file)
    file.close()


def deserialize(filename):
    file = open(filename, 'rb')
    result = pickle.load(file)
    file.close()
    return result

fig = plt.figure()
ax = fig.gca()
image_i = 0
def sim1resToImage(result):
    global image_i
    data = result.npVel
    obstacles = result.obstacles
    width = len(data)
    height = len(data[0])
    assert(width == len(obstacles))
    assert(height == len(obstacles[0]))
    assert(width == len(data))
    assert(height == len(data[0]))

    x, y = np.mgrid[0:width, 0:height]
    # Every 3rd arrow
    skip = (slice(None, None, 3), slice(None, None, 3))
    dx, dy, _ = np.transpose(data, (2, 0, 1))
    # Draw obstacles in the background
    obstacles = np.clip(np.reshape(obstacles, (width, height)), 0, 1)

    ax.set(aspect=1, title='Vector field')
    # ax.invert_yaxis()
    # ax.imshow(obstacles, interpolation='none', extent=[0, width, height, 0])
    ax.imshow(np.transpose(obstacles), interpolation='none')
    # ax.imshow(obstacles, interpolation='none')
    # ax.quiver(x[skip], y[skip], dx[skip], dy[skip])
    ax.quiver(np.transpose(dx), np.transpose(dy))
    # ax.quiver(dx, dy)

    ax.invert_yaxis()
    # fig.canvas.draw()
    # plt.show()
    ensureDir("images/")
    fig.savefig("images/fig_{}.png".format(image_i))
    image_i += 1

    # plt.pause(0.01)
    ax.clear()

def nn1resToImage(result):
    data = result.data
    scipy.ndimage.zoom(data, [2.0, 2.0, 1.0], order=1)
    ax, plt = arrToImage(data)
    plt.show()

def arrToImage(data):
    imageHeight = len(data)
    imageWidth = len(data[0])

    fig, ax = plt.subplots()
    x, y = np.mgrid[0:imageHeight, 0:imageWidth]
    # Every 3rd arrow
    skip = (slice(None, None, 3), slice(None, None, 3))
    dx, dy = np.transpose(data, (2, 0, 1))
    ax.quiver(x[skip], y[skip], dx[skip], dy[skip])
    ax.set(aspect=1, title='Vector field')
    return ax, plt
