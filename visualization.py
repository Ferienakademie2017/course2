import matplotlib.pyplot as plt
import numpy as np

VERBOSE = False
data = np.load("fluidSamples6432/0000.npy")
if VERBOSE:
    print(data.shape[0])
    print(data.shape[1])
    print(data.shape[2])

data_x = data[:, :, 0]
data_y = data[:, :, 1]
if VERBOSE:
    print(data_x.shape[0])
    print(data_x.shape[1])
    print(data_y.shape[0])
    print(data_y.shape[1])

x_axis = np.arange(data.shape[1])
y_axis = np.arange(data.shape[0])

fig, ax = plt.subplots()
q = ax.quiver(x_axis, y_axis, data_x, data_y)

plt.show()
