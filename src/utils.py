import os
import os.path
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage

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

image_i = 0
def sim1resToImage(result):
    global image_i, fig, ax
    try:
        fig
    except NameError:
        # Initialize figures
        fig = plt.figure()
        ax = fig.gca()

    data = result.npVel
    if data.shape[2] == 3:
        # Remove z coordinate
        data = np.delete(data, 2, 2)
    obstacles = result.obstacles
    widthMin = min(len(data), len(obstacles))
    heightMin = min(len(data[0]), len(obstacles[0]))
    widthMax = max(len(data), len(obstacles))
    heightMax = max(len(data[0]), len(obstacles[0]))
    # Fix for now to make it comparable
    widthMin = widthMax // 4
    heightMin = heightMax // 4

    dx, dy = np.transpose(data, (2, 0, 1))
    # Draw obstacles in the background
    obstacles = np.clip(obstacles, 0, 1)

    # Every n-th arrow
    skipData = (slice(None, None, len(data) // widthMin),
                slice(None, None, len(data[0]) // heightMin))
    skipCoord = (slice(None, None, len(obstacles) // widthMin),
                 slice(None, None, len(obstacles[0]) // heightMin))
    x, y = np.mgrid[0:widthMax, 0:heightMax]

    ax.set(aspect=1, title='Vector field')
    ax.imshow(obstacles, interpolation='none')
    ax.quiver(y[skipCoord], x[skipCoord], dx[skipData], dy[skipData])

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
        self.fig = plt.figure('Loss function')
        self.ax = self.fig.gca()
        #self.ax.set(title='Loss function')
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
            self.ax.set_xlim([0, max(500, len(self.x))])
            self.fig.canvas.draw()
            plt.pause(0.01)

    def save(self):
        self.ax.clear()
        self.ax.plot(self.x, self.y)
        self.ax.set_ylim([0, max(self.y)])
        self.ax.set_xlim([0, max(1000, len(self.x))])
        ensureDir("images/")
        self.fig.savefig("images/loss.png")
