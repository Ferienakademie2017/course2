import time
import os
import shutil
import sys
import math
import random

import tensorflow as tf
import numpy as np
import scipy.misc
np.random.seed(13)
tf.set_random_seed(13)

# path to fluid sim data
# fluidDataPath = "densitySamples1608/"
fluidDataPath = "densitySamples6432/"
fluidMetadataPath = "fluidSamplesMetadata/"

# path to trained models
trainedModelsPath = "trainedModels/"

# path to output
outputPath = "output/"

# training parameters
trainingEpochs = 1000

def twoDtoOneD(twoD):
	n_data = twoD.shape[0]
	return twoD.reshape([n_data, -1])

# load data
densities = []

# read y_positions in array
y_positions = np.load(fluidMetadataPath + "y_position_array.npy")
sample_count = y_positions.shape[0]

# read data from fluid sampling
for index in range(sample_count):
	density = np.load(fluidDataPath + "{:04d}.npy".format(index))
	densities.append(density)
densities = np.array(densities)

print("Read fluid data samples")

validationSize = int(sample_count * 0.1) # take 10% as validation samples

# desired output for validation and training
validationData = densities[sample_count-validationSize:sample_count][:]
trainingData = densities[0:sample_count-validationSize][:]

# input for validation and training
validationInput = y_positions[sample_count-validationSize:sample_count]
trainingInput = y_positions[0:sample_count-validationSize]

print("Split into %d training and %d validation samples" % (len(trainingData), len(validationData)) )

# set up network
sess = tf.Session()
input_layer = tf.placeholder(tf.float32, shape=(None, 1))
output_size = 1
for d in range(1, densities.ndim):
    output_size *= densities.shape[d]

W = tf.Variable(tf.random_normal([1, output_size], stddev=0.01))
b = tf.Variable(tf.random_normal([output_size], stddev=0.01))

output = input_layer * W + b

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(output - y)
loss = tf.reduce_sum(squared_deltas)

optimizer = tf.train.AdamOptimizer(0.0001)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)

flat_training_data = twoDtoOneD(trainingData)
trainingInput = trainingInput.reshape(-1, 1)
print(flat_training_data.shape)
print(trainingInput.shape)

for i in range(trainingEpochs):
	sess.run(train, feed_dict = {input_layer: trainingInput, y: flat_training_data})

print(sess.run([W, b]))

# test the trained network
test_output = sess.run(output, {input_layer: validationInput[0].reshape(1,1)})
formatted_test_output = test_output.reshape(densities.shape[1:])
np.save("test_output", formatted_test_output)
