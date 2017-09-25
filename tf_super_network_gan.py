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

# desired output for validation and training
validationData = velocities[sample_count-validationSize:sample_count][:] 
trainingData = velocities[0:sample_count-validationSize][:]

# input for validation and training
validationInput = y_positions[sample_count-validationSize:sample_count]
trainingInput = y_positions[0:sample_count-validationSize]

print("Split into %d training and %d validation samples" % (len(trainingData), len(validationData)) )

# from https://gist.github.com/wiseodd/b2697c620e39cb5b134bc6173cfe0f56
def xavier_init(size):
	in_dim = size[0]
	xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
	return tf.random_normal(shape=size, stddev=xavier_stddev)

# set up gan network
# Discriminator Net
X = tf.placeholder(tf.float32, shape=[None, 256], name='X')

D_W1 = tf.Variable(xavier_init([256, 128]), name='D_W1')
D_b1 = tf.Variable(tf.zeros(shape=[128]), name='D_b1')

D_W2 = tf.Variable(xavier_init([128, 1]), name='D_W2')
D_b2 = tf.Variable(tf.zeros(shape=[1]), name='D_b2')

theta_D = [D_W1, D_W2, D_b1, D_b2]

# Generator Net
Z = tf.placeholder(tf.float32, shape=[None, 64], name='Z')

G_W1 = tf.Variable(xavier_init([64, 128]), name='G_W1')
G_b1 = tf.Variable(tf.zeros(shape=[128]), name='G_b1')

G_W2 = tf.Variable(xavier_init([128, 256]), name='G_W2')
G_b2 = tf.Variable(tf.zeros(shape=[256]), name='G_b2')

theta_G = [G_W1, G_W2, G_b1, G_b2]


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit

# training
G_sample = generator(Z)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)

D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
G_loss = -tf.reduce_mean(tf.log(D_fake))

# Only update D(X)'s parameters, so var_list = theta_D
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
# Only update G(X)'s parameters, so var_list = theta_G
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

def sample_Z(m, n):
    '''Uniform prior for G(Z)'''
    return np.random.uniform(-1., 1., size=[m, n])

mb_size = 26
Z_dim = 64
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for it in range(25001):
    X_mb = twoDtoOneD(trainingData)

    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})
    if it % 100 == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'. format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()

for it in range(10):
	test_output = sess.run(G_sample, feed_dict={Z: sample_Z(1, Z_dim)})
	formatted_test_output = oneDtoTwoD(test_output)
	np.save("test_output{0}".format(it), formatted_test_output)

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