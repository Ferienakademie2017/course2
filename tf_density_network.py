from __future__ import division

import tensorflow as tf
import numpy as np
import os
import shutil
import os.path

from utils import get_parameter
from networks import create_1_256_network, create_1_8_256_network,\
        create_1_32_deconv_256_network

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

validation_indices = get_parameter("validation_indices")
training_indices = get_parameter("training_indices")
# desired output for validation and training
validationData = densities[validation_indices]
trainingData = densities[training_indices]

# input for validation and training, as Xx1 vector
validationInput = y_positions[validation_indices].reshape(-1, 1)
trainingInput = y_positions[training_indices].reshape(-1, 1)

training_batch_size = len(trainingData)
print("Split into %d training and %d validation samples" %
        (training_batch_size, len(validationData)))

print(trainingInput.shape)

# set up network
# input_layer, output = create_1_256_network(densities)
# input_layer, output = create_1_8_256_network(densities)
input_layer, output = create_1_32_deconv_256_network(
        data_shape=density_shape, batch_size=training_batch_size)

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(output - y)
loss = tf.reduce_sum(squared_deltas)

optimizer = tf.train.AdamOptimizer(0.0001)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(trainingEpochs):
    if i % 100 == 99:
        print("Training epoch {}...".format(i))
    sess.run(train,
            feed_dict={input_layer: trainingInput, y: trainingData})

# test the trained network
test_output = sess.run(output, {input_layer: validationInput})
formatted_test_output = test_output.reshape(
        (training_batch_size, ) + density_shape)

script_dir = os.path.dirname(os.path.abspath(__file__))
test_outputs_dir = os.path.join(script_dir, "testOutputs")
if os.path.isdir(test_outputs_dir):
    shutil.rmtree(test_outputs_dir)
os.makedirs(test_outputs_dir)

# output filename == validation index (visualization derives position of
# obstacle on that)
indices = list(range(densities.shape[0]))[validation_indices]
for i in range(formatted_test_output.shape[0]):
    np.save(os.path.join(
        test_outputs_dir, "{:04d}".format(indices[i])),
            formatted_test_output[i])
