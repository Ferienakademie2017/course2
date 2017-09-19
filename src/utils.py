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

def sim1resToImage(result, path):
    arrToImage(result.data, path)

def nn1resToImage(result, path):
    data = result.data
    scipy.ndimage.zoom(data, [2.0, 2.0, 1.0], order=1)
    arrToImage(data, path)

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
    plt.show()
