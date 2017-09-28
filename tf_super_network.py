#******************************************************************************
#
# MantaFlow fluid solver framework
# Copyright 2017 Nils Thuerey
#
# This program is free software, distributed under the terms of the
# GNU General Public License (GPL) 
# http://www.gnu.org/licenses
#
# As-simple-as-possible manta- & tensor-flow example
# Hard coded for 2d 64*64 density data
#
#******************************************************************************

#################
# CONFIGURATION #
#################

#for training just a single frame, set a number from 0 to Number of frames in "fluidSamples1608"
#for deactivating single frame training, set to -1
single_training_data = 10

#set the frame which is written to the test_output.npy
test_output_frame = 0


#Imports:
import time
import os
import shutil
import sys
import math
import random

import tensorflow as tf
import numpy as np
import scipy.misc

from networks import create_1_8_256_network
np.random.seed(13)
tf.set_random_seed(13)

# path to fluid sim data
fluidDataPath = "fluidSamples1608/"
fluidMetadataPath = "fluidSamplesMetadata/"

# path to trained models
trainedModelsPath = "trainedModels/"

# path to output
outputPath = "output/"

# training parameters
trainingEpochs = 2500
batchSize      = 10

# network parameters
inputHeight = 8
inputWidth  = 16
inSize      = inputHeight * inputWidth * 2 # warning - hard coded to scalar values 64^2


def twoDtoOneD(twoD):
	print (twoD.shape)
	n_data = twoD.shape[0]
	return twoD.reshape([n_data, -1])

def oneDtoTwoD(oneD):
	return oneD.reshape(8, 16, 2)

# load data
velocities = []

# read y_positions in array
y_positions = np.load(fluidMetadataPath + "y_position_array.npy")
sample_count = y_positions.shape[0]

# read data from fluid sampling
for index in range(sample_count):
	vel = np.load(fluidDataPath + "{:04d}.npy".format(index))
	velocities.append(vel)
velocities = np.array(velocities)

#densities = np.reshape( densities, (len(densities), 64,64,1) )

print("Read fluid data samples")

validationSize = int(sample_count * 0.1) # take 10% as validation samples
#print(str(velocities))


if single_training_data == -1:
	# desired output for validation and training
	validationData = velocities[sample_count-validationSize:sample_count][:] 
	trainingData = velocities[0:sample_count-validationSize][:]
	# input for validation and training
	validationInput = y_positions[sample_count-validationSize:sample_count]
	trainingInput = y_positions[0:sample_count-validationSize]
else:
	# desired output for validation and training
	trainingData = velocities[single_training_data:single_training_data+1][:]
	validationData = velocities[single_training_data:single_training_data+1][:]
	# input for validation and training
	validationInput = y_positions[single_training_data:single_training_data+1]
	trainingInput = y_positions[single_training_data:single_training_data+1]



print("Split into %d training and %d validation samples" % (len(trainingData), len(validationData)) )

# set up network
input_layer, output = create_1_8_256_network(velocities)

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(output - y)
loss = tf.reduce_sum(squared_deltas)

optimizer = tf.train.AdamOptimizer(0.0001)
train = optimizer.minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

#print(trainingInput)
#print(trainingData)
flat_training_data = twoDtoOneD(trainingData)
trainingInput = trainingInput.reshape(-1, 1)
print(flat_training_data.shape)
print(trainingInput.shape)

for i in range(1000):
	sess.run(train, feed_dict = {input_layer: trainingInput, y: flat_training_data})

# test the trained network

test_output = sess.run(output, {input_layer: trainingInput[test_output_frame].reshape(1,1)})
formatted_test_output = oneDtoTwoD(test_output)
np.save("test_output", formatted_test_output)
