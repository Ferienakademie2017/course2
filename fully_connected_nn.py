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
print(obstaclePos)
vel = np.reshape( velocities, (len(velocities), 32, 64, 2) )
pos = np.reshape( obstaclePos, (len(velocities), 1))
loadNum = velocities.shape[0]



insize=32*64*2 #np.size(vel) #number of entries
learning_rate=0.01
trainingEpochs=2000

#batches
batch_vel=velocities
batch_pos=obstaclePos

# Setting up the network
x = tf.placeholder(tf.float32, shape=[None, 1])
y = tf.placeholder(tf.float32, shape=[None, 64, 32, 2])

layer1=100
fc_1w = tf.Variable(tf.random_normal([1, layer1], stddev=0.01))
fc_1b   = tf.Variable(tf.random_normal([layer1], stddev=0.01))

fc1 = tf.add(tf.matmul(x, fc_1w), fc_1b)
#fc1 = tf.nn.relu(fc1)
fc1 = tf.nn.tanh(fc1)
#fc1 = tf.nn.dropout(fc1, 0.9) # plenty of dropout...

fc_2w = tf.Variable(tf.random_normal([layer1, inSize], stddev=0.01))  # back to input size
fc_2b = tf.Variable(tf.random_normal([inSize], stddev=0.01))

y_pred = tf.add(tf.matmul(fc1, fc_2w), fc_2b)
y_pred = tf.reshape( y_pred, shape=[-1, 64, 32, 2])


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
			plotfunction.plot_error(vali, out, "validation data point 16 - valdiation not normalized")
			#plotfunction_cost.plot_cost(epoch_steps,cost_data, 'cost_point16.pdf')










