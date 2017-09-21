import os
import os.path
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
    pickle.dump(obj, file, protocol=2)
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
    # assert(width == len(obstacles))
    # assert(height == len(obstacles[0]))
    assert(width == len(data))
    assert(height == len(data[0]))

    x, y = np.mgrid[0:width, 0:height]
    # Every 3rd arrow
    skip = (slice(None, None, 3), slice(None, None, 3))
    dx, dy, _ = np.transpose(data, (2, 0, 1))
    # Draw obstacles in the background
    obstacles = np.clip(obstacles, 0, 1)

    ax.set(aspect=1, title='Vector field')
    ax.imshow(obstacles, interpolation='none')
    # ax.quiver(x[skip], y[skip], dx[skip], dy[skip])
    ax.quiver(dx, dy)

    ax.invert_yaxis()
    # fig.canvas.draw()
    # plt.show()
    ensureDir("images/")
    fig.savefig("images/fig_{}.png".format(image_i))
    image_i += 1

    # plt.pause(0.01)
    ax.clear()

class LossLogger:
    def __init__(self, gui=True):
        # plt.ion()
        self.gui = gui
        self.fig = plt.figure()
        self.ax = self.fig.gca()
        self.x = []
        self.y = []

    def logLoss(self, i, loss):
        print("Loss: {}".format(loss))
        self.x.append(i)
        self.y.append(loss)
        if self.gui:
            self.ax.clear()
            self.ax.plot(self.x, self.y)
            self.ax.set_ylim([0, max(self.y)])
            self.ax.set_xlim([0, max(1000, len(self.x))])
            plt.pause(0.01)
