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
import plotscript_3D
np.random.seed(13)
tf.set_random_seed(13)

# data acquisition
vel = np.load('../mantaflow/manta/tensorflow/data/data.npy') #velocity field
pos = np.ones(1)        #position of the obstacle data input
vel_max = np.amax(vel) #max of velocity entries
vel = vel / vel_max #scaling of velocity

insize=np.size(vel) #number of entries
learning_rate=0.0001
trainingEpochs=10000

#batches
batch_vel=[]
batch_vel.append(vel)
batch_pos=[]
batch_pos.append(pos)
print(vel.shape, pos.shape, batch_pos)

#validation
vali_vel=[]
vali_vel.append(vel)
vali_pos=[]
vali_pos.append(pos)


# Setting up the network
x = tf.placeholder(tf.float32, shape=[None, 1])
y = tf.placeholder(tf.float32, shape=[None, 64, 32, 2])

fc_1w = tf.Variable(tf.random_normal([1, insize], stddev=0.01))
fc_1b   = tf.Variable(tf.random_normal([insize], stddev=0.01))

fc1 = tf.add(tf.matmul(x, fc_1w), fc_1b)

fc1 = tf.nn.tanh(fc1)

y_pred = tf.reshape( fc1, shape=[-1, 64, 32, 2])

cost = tf.nn.l2_loss(y - y_pred) 
opt  = tf.train.AdamOptimizer(learning_rate).minimize(cost) 	#gradient descent

#start training

print("Starting training...")
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for epoch in range(trainingEpochs):


	_ , currentCost = sess.run([opt, cost], feed_dict={x: batch_pos, y: batch_vel})
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
			
			vali = np.concatenate((vali, zeros), axis = 2)
			out = np.concatenate((out, zeros), axis = 2)
			
			scipy.misc.toimage( vali , cmin=0.0, cmax=1.0).save("in_%d.png" % i)
			scipy.misc.toimage( out , cmin=0.0, cmax=1.0).save("out_%d.png" % i)
			valiname="in_%d.pdf" % i
			outname="out_%d.pdf" % i
			plotscript_3D.plot(vali,valiname)
			plotscript_3D.plot(out,outname)










