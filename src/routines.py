import tensorflow as tf
import numpy as np
import csv
import math
import matplotlib.pyplot as plt


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def leaky_relu(x, alpha):
    return tf.maximum(x, alpha * x)

def to_image_form(data):
    return np.reshape(data, (32, 64, 2))


def load_data():
    training_image = []
    training_val = []
    for i in range(1, 32):
        if i == 16: continue
        path = "../res/karman_data_norm/vel" + str(i) + ".npy"
        training_image.append(np.load(path).flatten())
        training_val.append(i / 32.0)

    training_data = [np.reshape(training_val, (30, 1)), training_image]
    return training_data


def load_time_data():
    training_image = []
    training_val = []
    for i in range(100):
        path = "../res/timestep_norm/vel{}_{}.npy".format(str(16), str(i))
        training_image.append(np.load(path).flatten())
        training_val.append((16/32, i))

    training_data = [np.reshape(training_val, (100, 2)), training_image]
    return training_data


def get_scale_factor(y_pos):
    index = int(y_pos * 32 - 1)
    return np.load("../res/karman_data_norm/scale_factors.npy")[index]

def get_time_scale_factor(x):
    #index = int(x[] * 32 - 1)
    return np.load("../res/timestep_norm/scale_factors.npy")[x[1]]


def save_csv(data, path):
    with open(path, "w") as file:
        writer = csv.writer(file)
        for k, v in data.items():
            writer.writerow([k, v])


def plot(real_flow, net_flow):
    # takes ONE real flow and ONE output from network and compares them
    real_flow = real_flow.transpose((1, 0, 2))
    net_flow = net_flow.transpose((1, 0, 2))
    image_size = real_flow.shape

    skip = 2
    X, Y = np.mgrid[0:image_size[0]:skip, 0:image_size[1]:skip]

    [f, (ax1, ax2, ax3)] = plt.subplots(3, sharex=True, sharey=True)
    ax1.quiver(X, Y, real_flow[::skip, ::skip, 0], real_flow[::skip, ::skip, 1], units='inches')
    ax1.set_title("Real flow")
    ax1.set_xlim(0, image_size[0])
    ax1.set_ylim(0, image_size[1])
    ax2.set_title("Output of network")
    ax2.quiver(X, Y, net_flow[::skip, ::skip, 0], net_flow[::skip, ::skip, 1], units='inches')

    # compute error 
    diff_flow = (real_flow[:, :, 0] - net_flow[:, :, 0]) ** 2 + (real_flow[:, :, 1] - net_flow[:, :, 1]) ** 2
    diff_norm = math.sqrt(np.sum(diff_flow))

    real_flow_sq = real_flow[:, :, 0] ** 2 + real_flow[:, :, 1] ** 2
    real_norm = math.sqrt(np.sum(real_flow_sq))
    print("Average error: %f" % (diff_norm / real_norm))

    real_max = np.amax(real_flow_sq)
    diff_max = np.amax(diff_flow)

    ax3.set_title("Plot of velocity differences (real-net)")
    ax3.quiver(X, Y, real_flow[::skip, ::skip, 0] - net_flow[::skip, ::skip, 0],
               real_flow[::skip, ::skip, 1] - net_flow[::skip, ::skip, 1], scale=real_max, units='inches')

    plt.show()
    exit()
