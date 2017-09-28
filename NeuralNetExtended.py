import tensorflow as tf
import numpy as np
import scipy.misc
import readTrainingData
import random

np.random.seed(13)
tf.set_random_seed(13)

import flow

# script for first basic neural net where input layer
# (y coordinate of obstacle) is directly forwarded to output layer with 256 flow values

path_to_data = r'\course2\trainingData\trainingKarman32.p'
trainingEpochs = 1000
batchSize = 8
inSize = 1 # warning - hard coded to scalar values 1

# set up the network

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)  # training data

xIn = tf.reshape(x, shape=[-1, inSize])  # flatten
fc_1w = tf.Variable(tf.random_normal([inSize, 256], stddev=0.01))
fc_1b = tf.Variable(tf.random_normal([256], stddev=0.01))

fc_1 = tf.add(tf.matmul(xIn, fc_1w), fc_1b)
fc_1= tf.nn.tanh(fc_1)
# fc1 = tf.nn.dropout(fc1, 0.5)  # plenty of dropout...

fc_2w = tf.Variable(tf.random_normal([256, 256], stddev=0.01))  # back to input size
fc_2b = tf.Variable(tf.random_normal([256], stddev=0.01))

fc_2 = tf.add(tf.matmul(fc_1, fc_2w), fc_2b)
fc_2 = tf.nn.tanh(fc_2)
# y_pred = tf.add(tf.matmul(fc1, fc_2w), fc_2b)
# y_pred = tf.reshape(y_pred, shape=[-1, 64, 64, 1])
y_pred = fc_2
# y_pred = tf.reshape(y_pred, shape=[-1, 8, 16, 2])

cost = tf.nn.l2_loss((y - y_pred))/batchSize
opt = tf.train.GradientDescentOptimizer(0.2).minimize(cost)

# now we can start training...

# read input  and training data
position_y, training_data = readTrainingData.loadData(path_to_data)
# training_data = np.reshape(training_data, newshape=[-1, 8, 16, 2])
scaling = np.ndarray.max(abs(training_data))
training_data = training_data/scaling
loadNum = len(position_y)

print("Starting training...")
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for epoch in range(trainingEpochs):
    #c = (epoch * batchSize) % training_data.shape[0]
    batch_training_out = []
    batch_training_in = []
    for currNo in range(0, batchSize):
        r = random.randint(0, loadNum - 1)
        batch_training_out.append(training_data[r, :])
        batch_training_in.append(position_y[r])
    #batch_xs, batch_ys = training_data[c:c+batchSize,:], training_data[c:c+batchSize,:]

    _, currentCost = sess.run([opt, cost], feed_dict={x: batch_training_in, y: batch_training_out})
    print("Epoch %d/%d: cost %f " % (epoch + 1, trainingEpochs, currentCost) )

    if epoch == trainingEpochs - 1:
        [valiCost, vout] = sess.run([cost, y_pred], feed_dict={x: position_y[0], y: training_data[6,:]})
        print(" Validation: cost %f " % (valiCost))

        if epoch == trainingEpochs - 1:
            # for i in range(len(training_data)):
                valiData = readTrainingData.transformToImage(training_data[6,:], [8, 16, 2])
                vout_img = readTrainingData.transformToImage(vout, [8, 16, 2])
                # scipy.misc.toimage(valiData[:,:,0], cmin=0.0, cmax=1.0).save("inx_%d.png" % i)
                # scipy.misc.toimage(valiData[:, :, 1], cmin=0.0, cmax=1.0).save("iny_%d.png" % i)
                # scipy.misc.toimage(vout_img[:, :, 0], cmin=0.0, cmax=1.0).save("outx_%d.png" % i)
                # scipy.misc.toimage(vout_img[:, :, 1], cmin=0.0, cmax=1.0).save("outy_%d.png" % i)
                flow.plot_flow_triple(valiData, vout_img)

print("Done")