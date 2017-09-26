import time
import os
import shutil
import sys
import math
import random

import tensorflow as tf
import numpy as np
import scipy.misc

from networks import create_1_256_network, create_1_8_256_network

np.random.seed(13)
tf.set_random_seed(13)

# path to fluid sim data
# fluidDataPath = "densitySamples1608/"
fluidDataPath = "densitySamples6432/"
fluidMetadataPath = "fluidSamplesMetadata/"

# path to trained models
trainedModelsPath = "trainedModels/"

# training parameters
trainingEpochs = 1000

# load data
densities = []

# read y_positions in array
y_positions = np.load(fluidMetadataPath + "y_position_array.npy")
sample_count = y_positions.shape[0]

density_shape = None
# read data from fluid sampling
# 2D data is flattened but the shape is remembered for test_output
for index in range(sample_count):
    density = np.load(fluidDataPath + "{:04d}.npy".format(index))
    if density_shape is None:
        density_shape = density.shape
    densities.append(density.flatten())
densities = np.array(densities)

print("Read fluid data samples")

validationSize = int(sample_count * 0.1) # take 10% as validation samples
validation_start_index = 14
validation_end_index = validation_start_index + validationSize
print("Validation data range {}:{}".format(
    validation_start_index, validation_end_index))

# desired output for validation and training
validationData = densities[validation_start_index:validation_end_index][:]
trainingData = np.vstack((
    densities[0:validation_start_index][:],
    densities[validation_end_index:][:]
    ))

# input for validation and training
validationInput = y_positions[validation_start_index:validation_end_index]
trainingInput = np.hstack((
    y_positions[:validation_start_index][:],
    y_positions[validation_end_index:][:]
    ))

print("Split into %d training and %d validation samples" %
        (len(trainingData), len(validationData)))

# set up network
input_layer, output = create_1_256_network(densities)
# input_layer, output = create_1_8_256_network(densities)

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(output - y)
loss = tf.reduce_sum(squared_deltas)

optimizer = tf.train.AdamOptimizer(0.0001)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

flat_training_data = trainingData
trainingInput = trainingInput.reshape(-1, 1)
print(flat_training_data.shape)
print(trainingInput.shape)

for i in range(trainingEpochs):
	sess.run(train, feed_dict={input_layer: trainingInput, y: flat_training_data})

# test the trained network
test_output = sess.run(output, {input_layer: validationInput[0].reshape(1,1)})
formatted_test_output = test_output.reshape(density_shape)
np.save("test_output", formatted_test_output)
