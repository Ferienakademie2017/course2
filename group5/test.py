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
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
import scipy.misc
np.random.seed(13)
tf.set_random_seed(13)

sys.path.append("../tools")
import uniio

# path to sim data, trained models and output are also saved here
basePath = '.'

trainingEpochs = 50000
inSize         = 64 * 32 * 3 # warning - hard coded to scalar values 64^2

# load data
velocities = []
obstaclePos = []

# start reading simSimple 1000 ff.
#for sim in range(1000,2000): 
	# if os.path.exists( "%s/simSimple_%04d" % (basePath, sim) ):
tmp = 0
for i in range(0,101): 
	filename = "%s/data_%s.npy"
	
	uniPath = filename % (basePath, str(i))  # 100 files per sim
	
	
	content = np.load(uniPath)
	

	h = 32
	w = 64
	

	velocities.append( content )
	obstaclePos.append([tmp])
	tmp += 1





loadNum = len(velocities)
# if loadNum<200:
# 	print("Error - use at least two full sims, generate data by running 'manta ./manta_genSimSimple.py' a few times..."); exit(1)

velocities = np.reshape( velocities, (len(velocities), 64, 32, 1, 3) )
obstaclePos = np.reshape( obstaclePos, (len(velocities), 1))
print(obstaclePos)

# print("Read uni files, total data " + format(velocities.shape) )
# valiSize = int(loadNum * 0.1) # at least 1 full sim...
# valiData = velocities[loadNum-valiSize:loadNum,:] 
# velocities = velocities[0:loadNum-valiSize,:]
# print("Split into %d training and %d validation samples" % (velocities.shape[0], valiData.shape[0]) )
loadNum = velocities.shape[0]






# set up the network

x = tf.placeholder(tf.float32, shape=[None, 1])
y = tf.placeholder(tf.float32, shape=[None, 64, 32, 1, 3])

# xIn = tf.reshape(x, shape=[-1, inSize ]) # flatten
fc_1w = tf.Variable(tf.random_normal([1, 5000], stddev=0.01))
fc_1b   = tf.Variable(tf.random_normal([5000], stddev=0.01))

fc1 = tf.add(tf.matmul(x, fc_1w), fc_1b)
fc1 = tf.nn.tanh(fc1)
# fc1 = tf.nn.dropout(fc1, 0.9) # plenty of dropout...

fc_2w = tf.Variable(tf.random_normal([5000, inSize], stddev=0.01))  # back to input size
fc_2b = tf.Variable(tf.random_normal([inSize], stddev=0.01))

y_pred = tf.add(tf.matmul(fc1, fc_2w), fc_2b)
y_pred = tf.reshape( y_pred, shape=[-1, 64, 32, 1, 3])

cost = tf.nn.l2_loss(y - y_pred) 
opt  = tf.train.AdamOptimizer(0.0001).minimize(cost)

# now we can start training...

print("Starting training...")
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for epoch in range(trainingEpochs):
	
	
	#batch_xs, batch_ys = densities[c:c+batchSize,:], densities[c:c+batchSize,:]
	_ , currentCost = sess.run([opt, cost], feed_dict={x: obstaclePos, y: velocities})
	print("Epoch %d/%d: cost %f " % (epoch, trainingEpochs, currentCost) )
	
	# if epoch%10==9 or epoch==trainingEpochs-1:
	# 	[valiCost,vout] = sess.run([cost, y_pred], feed_dict={x: valiData, y: valiData})
	# 	print("Epoch %d/%d: cost %f , validation cost %f " % (epoch, trainingEpochs, currentCost, valiCost) )
	# 	#print("Validation cost %f " % (valiCost) )

	# 	if epoch==trainingEpochs-1:
	# 		for i in range(len(valiData)):
	# 			scipy.misc.toimage( np.reshape(valiData[i], [64, 32, 1, 3]) , cmin=0.0, cmax=1.0).save("in_%d.png" % i)
	# 			scipy.misc.toimage( np.reshape(vout[i]    , [64, 32, 1, 3]) , cmin=0.0, cmax=1.0).save("out_%d.png" % i)



print("Done")