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
single_training_data = -1

#set the frame which is written to the test_output.npy
test_output_frame = 4

#number of evaluation data (between 0 and 1) and test data
validation_rate = 0.3





##
#Prevent misconfiguration:
if single_training_data != -1:
	test_output_frame=0


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

#NETWORK ARCHITECTURE:
#   Input        Layer1         Layer2          Layer3        Output-Layer
#   1 (lin)  ->  256 (tanh) ->  512 (tanh)  ->  256 (tanh) ->  256 (lin)
input_size = 1
layer1_size = 256
layer2_size = 512
layer3_size = 256
output_size = 256


sess = tf.Session()

input_layer = tf.placeholder(tf.float32, shape=(None, 1))

W1 = tf.Variable(tf.random_normal([1, layer1_size], stddev=0.01))
b1 = tf.Variable(tf.random_normal([layer1_size], stddev=0.01))
layer1 = tf.matmul(input_layer, W1) + b1
layer1 = tf.tanh(layer1)

W2 = tf.Variable(tf.random_normal([layer1_size, layer2_size], stddev=0.01))
b2 = tf.Variable(tf.random_normal([layer2_size], stddev=0.01))
layer2 = tf.matmul(layer1, W2) + b2
layer2 = tf.tanh(layer2)

W3 = tf.Variable(tf.random_normal([layer2_size, layer3_size], stddev=0.01))
b3 = tf.Variable(tf.random_normal([layer3_size], stddev=0.01))
layer3 = tf.matmul(layer2, W3) + b3
layer3 = tf.tanh(layer3)

W4 = tf.Variable(tf.random_normal([layer3_size, output_size], stddev=0.01))
b4 = tf.Variable(tf.random_normal([output_size], stddev=0.01))
layer4 = tf.matmul(layer3, W4) + b4

output = layer4


y = tf.placeholder(tf.float32)
squared_deltas = tf.square(output - y)
loss = tf.reduce_sum(squared_deltas)


optimizer = tf.train.AdamOptimizer(0.0001)
global_step = tf.Variable(0, name='global_step', trainable=False)
train = optimizer.minimize(loss, global_step=global_step)

init = tf.global_variables_initializer()
sess.run(init)

#print(trainingInput)
#print(trainingData)
flat_training_data = twoDtoOneD(trainingData)
trainingInput = trainingInput.reshape(-1, 1)
flat_validation_data = twoDtoOneD(validationData)
validationInput = validationInput.reshape(-1, 1)

print(flat_training_data.shape)
print(trainingInput.shape)

#create a summary
summary = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter("training", sess.graph)
tf.summary.scalar('loss', loss)


for i in range(3):
	for j in range(1000):
		loss=sess.run(train, feed_dict = {input_layer: trainingInput, y: flat_training_data})
		#summary_str = sess.run(summary, feed_dict={input_layer: trainingInput, y: flat_training_data})
		#summary_writer.add_summary(summary_str, step)
	#loss=sess.run(train, feed_dict = {input_layer: validationInput, y: flat_validation_data}, is_training=False)

#print(sess.run([W1, b1]))

# test the trained network

test_output = sess.run(output, {input_layer: trainingInput[test_output_frame].reshape(1,1)})
formatted_test_output = oneDtoTwoD(test_output)
np.save("test_output", formatted_test_output)


# set up the network
#
#x = tf.placeholder(tf.float32, shape=[None, 64,64, 1])
#y = tf.placeholder(tf.float32, shape=[None, 64,64, 1])
#
#xIn = tf.reshape(x, shape=[-1, inSize ]) # flatten
#fc_1w = tf.Variable(tf.random_normal([inSize, 50], stddev=0.01))
#fc_1b   = tf.Variable(tf.random_normal([50], stddev=0.01))
#
#fc1 = tf.add(tf.matmul(xIn, fc_1w), fc_1b)
#fc1 = tf.nn.tanh(fc1)
#fc1 = tf.nn.dropout(fc1, 0.5) # plenty of dropout...
#
#fc_2w = tf.Variable(tf.random_normal([50, inSize], stddev=0.01))  # back to input size
#fc_2b = tf.Variable(tf.random_normal([inSize], stddev=0.01))
#
#y_pred = tf.add(tf.matmul(fc1, fc_2w), fc_2b)
#y_pred = tf.reshape( y_pred, shape=[-1, 64, 64, 1])
#
#cost = tf.nn.l2_loss(y - y_pred) 
#opt  = tf.train.AdamOptimizer(0.0001).minimize(cost)
#


# now we can start training...

#print("Starting training...")
#sess = tf.InteractiveSession()
#sess.run(tf.global_variables_initializer())
#
#for epoch in range(trainingEpochs):
#	c = (epoch * batchSize) % densities.shape[0]
#	batch = []
#	for currNo in range(0, batchSize):
#		r = random.randint(0, loadNum-1) 
#		batch.append( densities[r] )
#
#	_ , currentCost = sess.run([opt, cost], feed_dict={x: batch, y: batch})
#	#print("Epoch %d/%d: cost %f " % (epoch, trainingEpochs, currentCost) ) # debug, always output cost
#	
#	if epoch%10==9 or epoch==trainingEpochs-1:
#		[valiCost,vout] = sess.run([cost, y_pred], feed_dict={x: valiData, y: valiData})
#		print("Epoch %d/%d: cost %f , validation cost %f " % (epoch, trainingEpochs, currentCost, valiCost) )
#
#		if epoch==trainingEpochs-1:
#			print("\n Training done. Writing %d images from validation data to current directory..." % len(valiData) )
#			for i in range(len(valiData)):
#				scipy.misc.toimage( np.reshape(valiData[i], [64, 64]) , cmin=0.0, cmax=1.0).save("in_%d.png" % i)
#				scipy.misc.toimage( np.reshape(vout[i]    , [64, 64]) , cmin=0.0, cmax=1.0).save("out_%d.png" % i)
#
#
#
#print("Done")
#
#
