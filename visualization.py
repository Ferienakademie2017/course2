import matplotlib.pyplot as plt
import numpy as np

data = np.load("fluidSamples6432/0000.npy")
print(data)
print(data.shape[0])
print(data.shape[1])
print(data.shape[2])
#print(refined_data)
data_x = data[0,:,:]
data_y = data[1,:,:]
data_x = data_x.transpose()
data_y = data_y.transpose()
print(data_x)
print(data_x.shape[0])
print(data_x.shape[1])
print(data_y)
print(data_y.shape[0])
print(data_y.shape[1])
#print(data_y)
x_axis = np.arange(0, 32, 1)
print(x_axis)
y_axis = np.arange(0, 64, 1)
print(y_axis)
#x, y = np.mgrid[0:16, 0:8]

fig, ax = plt.subplots()
q = ax.quiver(x_axis, y_axis, data_x, data_y)

plt.show()