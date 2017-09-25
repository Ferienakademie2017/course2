
# coding: utf-8

# In[2]:

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


# In[24]:

inSize = 64 * 32 * 2
nTrainingData = 101
velocities = []
obstaclePos = []


# In[25]:

# reading example images
tmp = 0
for i in range(nTrainingData):
    filename = "./vel_%s.npy"
    path = filename % (str(i))
    content = np.load(path)
    velocities.append(content)
    obstaclePos.append([tmp])
    tmp += 1
    
velocities = velocities / np.amax(velocities)


# In[26]:

loadNum = len(velocities)
velocities = np.reshape( velocities, (len(velocities), 32, 64, 2) )
obstaclePos = np.reshape( obstaclePos, (len(velocities), 1))
loadNum = velocities.shape[0]


# In[27]:

# full connected neural network
with tf.device('/gpu:0'):
    x = tf.placeholder(tf.float32, shape=[None, 1])
    y = tf.placeholder(tf.float32, shape=[None, 32, 64, 2])

    # xIn = tf.reshape(x, shape=[-1, inSize ]) # flatten
    fc_1w = tf.Variable(tf.random_normal([1, 256], stddev=0.01))
    fc_1b   = tf.Variable(tf.random_normal([256], stddev=0.01))

    fc1 = tf.add(tf.matmul(x, fc_1w), fc_1b)
    fc1 = tf.nn.relu(fc1)
    # fc1 = tf.nn.dropout(fc1, 0.9) # plenty of dropout...

    fc_2w = tf.Variable(tf.random_normal([256, inSize], stddev=0.01))  # back to input size
    fc_2b = tf.Variable(tf.random_normal([inSize], stddev=0.01))

    y_pred = tf.add(tf.matmul(fc1, fc_2w), fc_2b)
    y_pred = tf.reshape( y_pred, shape=[-1, 32, 64, 2])

    cost = tf.nn.l2_loss(y - y_pred) 
    opt  = tf.train.AdamOptimizer(0.001).minimize(cost)
    
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    trainingEpochs = 50000
    for epoch in range(trainingEpochs):    
        _ , currentCost = sess.run([opt, cost], feed_dict={x: obstaclePos, y: velocities})
        if(epoch%200 is 0):
            print("Epoch %d/%d: cost %f " % (epoch, trainingEpochs, currentCost) )
    print("Done")


# In[28]:

from scipy import misc
img = sess.run(y_pred, feed_dict={x: np.reshape([50], (1, 1))})
# misc.imsave("./test_data.png", img)
img = img[0]


# In[29]:

get_ipython().magic('matplotlib inline')
# predicted image
X, Y = np.meshgrid(np.arange(0, 64, 1), np.arange(0, 32, 1))
x_part = img[:, :, 0]
y_part = img[:, :, 1]
Q = plt.quiver(X, Y, x_part, y_part, units='width')
plt.figure()


# In[22]:

# input image
X, Y = np.meshgrid(np.arange(0, 64, 1), np.arange(0, 32, 1))
x_part = velocities[50, :, :, 0]
y_part = velocities[50, :, :, 1]
Q = plt.quiver(X, Y, x_part, y_part, units='width')
plt.figure()


# In[ ]:



