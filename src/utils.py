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

image_i = 0
def sim1resToImage(result, background='obstacles', origRes=None, folder=None):
    global image_i, fig, ax
    try:
        fig
    except NameError:
        # Initialize figures
        fig = plt.figure()
        ax = fig.gca()
    if folder == None:
        folder = "images/"
    cb = None

    data = result.npVel
    if data.shape[2] == 3:
        # Remove z coordinate
        data = np.delete(data, 2, 2)
    data = np.transpose(data, (1, 0, 2))
    obstacles = np.transpose(result.obstacles)
    widthMin = min(len(data), len(obstacles))
    heightMin = min(len(data[0]), len(obstacles[0]))
    widthMax = max(len(data), len(obstacles))
    heightMax = max(len(data[0]), len(obstacles[0]))
    # Fix for now to make it comparable
    # widthMin = widthMax // 4
    # heightMin = heightMax // 4

    # print(data.shape)

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

    if background == 'obstacles':
        ax.imshow(obstacles, interpolation='none')
        ax.quiver(y[skipCoord], x[skipCoord], dx[skipData], dy[skipData], scale=widthMin / widthMax, scale_units='x') # scale = widthMin / widthMax
    elif background == 'error':
        orig = origRes.npVel
        if orig.shape[2] == 3:
            # Remove z coordinate
            orig = np.delete(orig, 2, 2)
        orig = np.transpose(orig, (1, 0, 2))
        diff = orig - data
        # Remove diff at obstacles
        func = lambda x: 1.0 if x > 0.0001 else 0.0 #todo
        flagField = np.vectorize(func)(obstacles)
        diffBackg = flagField * np.sum(np.abs(diff), -1)

        diffAx = ax.imshow(diffBackg, interpolation='none')
        cb = fig.colorbar(diffAx)

        dx, dy = np.transpose(diff, (2, 0, 1))
        ax.quiver(y[skipCoord], x[skipCoord], dx[skipData], dy[skipData], scale=widthMin / widthMax, scale_units='x') # scale = widthMin / widthMax

    ax.invert_yaxis()
    # fig.canvas.draw()
    # plt.show()
    ensureDir(folder)
    fig.savefig("{}fig_{}.png".format(folder, image_i))
    image_i += 1

    # plt.pause(0.01)
    if cb != None:
        cb.remove()
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

    def plot(self):
        self.ax.clear()
        self.ax.semilogy(self.x, self.y)
        self.ax.set_ylim([0, max(self.y)])
        self.ax.set_xlim([0, max(2000, max(self.x))])

    def logLoss(self, i, loss):
        print("Loss: {}".format(loss))
        self.x.append(i)
        self.y.append(loss)
        if self.gui:
            self.plot()
            self.fig.canvas.draw()
            plt.pause(0.01)

    def save(self):
        self.plot()
        ensureDir("images/")
        self.fig.savefig("images/loss.png")
