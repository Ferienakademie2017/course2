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

mb_size = 5
Z_dim = 128

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
	picture_number = int(index*100)
	vel = np.load(fluidDataPath + "{:04d}.npy".format(picture_number))
	velocities.append(vel)
velocities = np.array(velocities)

#densities = np.reshape( densities, (len(densities), 64,64,1) )

print("Read fluid data samples")

validationSize = int(sample_count * 0) # take 10% as validation samples
#print(str(velocities))

# desired output for validation and training
validationData = velocities[sample_count-validationSize:sample_count][:] 
trainingData = velocities[0:sample_count-validationSize][:]

# input for validation and training
#validationInput = y_positions[sample_count-validationSize:sample_count]
#trainingInput = y_positions[0:sample_count-validationSize]

print("Split into %d training and %d validation samples" % (len(trainingData), len(validationData)) )

# from https://gist.github.com/wiseodd/b2697c620e39cb5b134bc6173cfe0f56
def xavier_init(size):
	in_dim = size[0]
	xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
	return tf.random_normal(shape=size, stddev=xavier_stddev)

def get_random_batch(array, batch_size):
	np.random.shuffle(array)
	return array[:batch_size]

# set up gan network

# Generator Net
Z = tf.placeholder(tf.float32, shape=[None, 128], name='Z')

G_W1 = tf.Variable(xavier_init([128, 128]), name='G_W1')
G_b1 = tf.Variable(tf.random_normal(shape=[128]), name='G_b1')

G_W1_5 = tf.Variable(xavier_init([128, 256]), name='G_W1_5')
G_b1_5 = tf.Variable(tf.zeros(shape=[256]), name='G_b1_5')

G_W2 = tf.Variable(xavier_init([256, 256]), name='G_W2')
G_b2 = tf.Variable(tf.random_normal(shape=[256]), name='G_b2')

theta_G = [G_W1, G_W1_5, G_W2, G_b1, G_b1_5, G_b2]

input_layer = tf.placeholder(tf.float32, shape=[mb_size, 8, 16, 2], name='X')

def generator(z):
    G_h1 = tf.nn.softplus(tf.matmul(z, G_W1) + G_b1)
    G_h1_5 = tf.nn.softplus(tf.matmul(G_h1, G_W1_5) + G_b1_5)
    G_log_prob = tf.matmul(G_h1_5, G_W2) + G_b2
    G_prob = G_log_prob * 0.25
    return G_prob


def discriminator(x):
    D_conv1 = tf.layers.conv2d(x, 32, [5, 5], activation=tf.nn.relu)
    D_conv1_flat = tf.reshape(D_conv1, [-1, 12*4*32])
    D_dense = tf.layers.dense(D_conv1_flat, units=256, activation=tf.nn.relu)
    D_dense2 = tf.layers.dense(D_dense, 1, tf.nn.sigmoid)

    return D_dense2

# training
G_sample = generator(Z)
D_real = discriminator(input_layer)
D_fake = discriminator(tf.reshape(G_sample, shape=(-1, 8, 16, 2)))

D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
G_loss = -tf.reduce_mean(tf.log(D_fake))

# Only update D(X)'s parameters, so var_list = theta_D
D_solver = tf.train.AdamOptimizer().minimize(D_loss) #var_list=theta_D)
# Only update G(X)'s parameters, so var_list = theta_G
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

def sample_Z(m, n):
    '''Uniform prior for G(Z)'''
    return np.random.uniform(-0.1, .1, size=[m, n])

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for it in range(2001):
    X_mb = get_random_batch(trainingData, mb_size)

    for i in range(2):
    	_, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={input_layer: X_mb, Z: sample_Z(mb_size, Z_dim)})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})
    if it % 100 == 0:
        print('Iter: {}'.format(it))
        print('D_loss: {:.4}'. format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()

for it in range(2):
	test_output = sess.run(G_sample, feed_dict={Z: sample_Z(1, Z_dim)})
	formatted_test_output = oneDtoTwoD(test_output)
	np.save("test_output{0}".format(it), formatted_test_output)

