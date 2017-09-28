#Simple single layer network using one velocity field as training data

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
import plotfunction
import plotfunction_cost
np.random.seed(13)
tf.set_random_seed(13)


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def deconv_layer(x, W_shape, b_shape, name, padding='SAME'):
	W = weight_variable(W_shape)
	b = bias_variable([b_shape])

	x_shape = tf.shape(x)
	out_shape = tf.stack([x_shape[0], x_shape[1], x_shape[2], W_shape[2]])

	return tf.nn.conv2d_transpose(x, W, out_shape, [1, 1, 1, 1], padding=padding) + b

basePath='../sim_data/data_obstacle_pos'
#	data acquisition
#	start reading data 0-31

inSize = 64 * 32 * 2
nTrainingData = 31
vali_num=16
velocities = []
obstaclePos = []


# reading example images
tmp = 0
for i in range(0,16):
    filename = "../sim_data/data_obstacle_pos/data%s.npy"
    path = filename % (str(i))
    content = np.load(path)
    velocities.append(content)
    obstaclePos.append([tmp])
    tmp += 1

tmp=17
for i in range(17,32):
    filename = "../sim_data/data_obstacle_pos/data%s.npy"
    path = filename % (str(i))
    content = np.load(path)
    velocities.append(content)
    obstaclePos.append([tmp])
    tmp += 1 

  
#validation
vali_vel=[]
filename_vali = "../sim_data/data_obstacle_pos/data%s.npy"
path_vali = filename_vali % (str(vali_num))
all_vel = list(velocities)
all_vel.append(np.load(path_vali))
max_vel = np.amax(all_vel)

vali_vel.append(np.load(path_vali)/max_vel)
vali_pos=[]
vali_pos.append([vali_num])

velocities = velocities / max_vel

loadNum = len(velocities)
vel = np.reshape( velocities, (len(velocities), 32, 64, 2) )
pos = np.reshape( obstaclePos, (len(velocities), 1))
loadNum = velocities.shape[0]



insize=32*64*2 #np.size(vel) #number of entries
learning_rate=0.001
trainingEpochs=2000

#batches
batch_vel=velocities
batch_pos=obstaclePos

# Setting up the network
x = tf.placeholder(tf.float32, shape=[None, 1])
y = tf.placeholder(tf.float32, shape=[None, 64, 32, 2])

#deconvolution
#tf.nn.conv2d_transpose(value, filter, output_shape, strides, padding, name)
#with tf.name_scope('deconv') as scope:
#	deconv = tf.nn.conv2d_transpose(input_layer, [3, 3, 1, 1], [1, 26, 20, 1], [1, 2, 2, 1], padding='SAME', name=None)

layer1=32*32
fc_1w = tf.Variable(tf.random_normal([1, layer1], stddev=0.01))
fc_1b   = tf.Variable(tf.random_normal([layer1], stddev=0.01))

fc1 = tf.add(tf.matmul(x, fc_1w), fc_1b)
fc1 = tf.nn.relu(fc1)
#fc1 = tf.nn.tanh(fc1)
#fc1 = tf.nn.dropout(fc1, 0.9) # plenty of dropout...

#fc1 = np.reshape(fc1, [-1, 8, 16, 16])

# Input Layer
input_layer = tf.reshape(fc1, [-1, 32, 32, 1])

# Convolutional Layer #1
conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

# Pooling Layer #1
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

# Convolutional Layer #2 and Pooling Layer #2
conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

# Dense Layer
pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64])
#dense = tf.layers.dense(inputs=pool2_flat, units=4096, activation=tf.nn.relu)
dropout = tf.layers.dropout(inputs=pool2_flat, rate=0.4)#, training=mode == tf.estimator.ModeKeys.TRAIN)







#deconv_1w = tf.Variable(tf.truncated_normal([3,3, 16, 8], stddev=0.01))
#deconv_1b = tf.Variable(tf.random_normal([inSize], stddev=0.01))
#deconv = deconv_layer(fc1, [3, 3, 512, 512], 512, 'deconv_5_3')
#deconv1 = tf.nn.conv2d_transpose(fc1,deconv_1w,[8,8,16],[1,1,1,1], padding='same') #+deconv_1b
#deconv1 = tf.image.resize_images(deconv1,[8,16,32])
#fc_2w = tf.Variable(tf.random_normal([layer1, inSize], stddev=0.01))  # back to input size
#fc_2b = tf.Variable(tf.random_normal([inSize], stddev=0.01))

#y_pred = tf.add(tf.matmul(fc1, fc_2w), fc_2b)
y_pred = tf.reshape( dropout, shape=[-1, 64, 32, 2])


cost = tf.nn.l2_loss(y - y_pred) 
opt  = tf.train.AdamOptimizer(learning_rate).minimize(cost) 	#gradient descent

#start training

print("Starting training...")
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
epoch_steps=[]
cost_data=[]
for epoch in range(trainingEpochs):
	if epoch==1000:
		learning_rate=0.0001
		

	_ , currentCost = sess.run([opt, cost], feed_dict={x: batch_pos, y: batch_vel})
	epoch_steps.append([epoch])
	cost_data.append([currentCost])
	print("Epoch %d/%d: cost %f " % (epoch, trainingEpochs, currentCost) )


	if epoch==trainingEpochs-1:
		[valiCost,vout] = sess.run([cost, y_pred], feed_dict={x: vali_pos, y: vali_vel})
		print("Epoch %d/%d: cost %f , validation cost %f " % (epoch, trainingEpochs, currentCost, valiCost) )

		print("\n Training done. Writing %d images from validation data to current directory..." % len(vali_pos) )
		print(vout.shape)
		for i in range(len(vali_pos)):
			zeros = np.zeros((64,32,1))
			vali = np.reshape(vali_vel[i], [64, 32,2])
			out = np.reshape(vout[i]    , [64, 32,2])
			plot_path='/home/anne/ferienakademie/eva_data/obstacle_pos/'
			valiname="in_%d" % i
			outname="out_%d" % i
			plotfunction.plot(vali,valiname)
			plotfunction.plot(out,outname)
			plotfunction.plot_error(vali, out,"", "convolution")
			#plotfunction_cost.plot_cost(epoch_steps,cost_data, 'cost_point16.pdf')








