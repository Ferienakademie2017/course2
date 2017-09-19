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
fluidDataPath = "fluidSamples/"

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

# load data
velocities = []

# read y_positions in array
y_positions = np.load(fluidDataPath + "y_position_array.npy")
sample_count = y_positions.shape[0]

# read data from fluid sampling
for index in range(sample_count):
	vel = np.load(fluidDataPath + "{:03d}.npy".format(index))
	velocities.append(vel)

#densities = np.reshape( densities, (len(densities), 64,64,1) )

print("Read fluid data samples")

validationSize = int(sample_count * 0.1) # take 10% as validation samples
#print(str(velocities))
validationData = velocities[sample_count-validationSize:sample_count][:] 
trainingData = velocities[0:sample_count-validationSize][:]
print("Split into %d training and %d validation samples" % (len(trainingData), len(validationData)) )



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