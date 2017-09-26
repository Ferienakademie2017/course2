import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
import scipy.misc
import readTrainingData
import random
import matplotlib.pyplot as plt

np.random.seed(13)
tf.set_random_seed(13)

import flow

# script for first basic neural net where input layer
# (y coordinate of obstacle) is directly forwarded to output layer with 256 flow values

path_to_data = r'C:\Users\Annika\Saved Games\Desktop\course2\trainingData\trainingKarman32_1000randu.p'
# path_to_data = r'C:\Users\Nico\Documents\Ferienakademie\course2\trainingData\trainingKarman32i100.p'
# path_to_data = r'C:\Users\Nico\Documents\Ferienakademie\course2\trainingData\trainingKarman32.p'
#path_to_data = r'C:\Users\Nico\Documents\Ferienakademie\course2\trainingData\trainingKarman32_1000randu.p'
trainingEpochs = 500
batchSize = 128
inSize = 1  # warning - hard coded to scalar values 1
validationProportion = 0.05
learningRate = 0.0005
error = []

# set up the network

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)  # training data

xIn = tf.reshape(x, shape=[-1, inSize])  # flatten
size1 = 64
size2 = 128
size3 = 256

mode = False

# fc_1 = layers.fully_connected(xIn,size1,activation_fn=tf.contrib.keras.layers.LeakyReLU(0.2))
# fc_2 = layers.fully_connected(fc_1,size2,activation_fn=tf.contrib.keras.layers.LeakyReLU(0.2))
fc_3 = layers.fully_connected(xIn,size3,activation_fn = tf.nn.tanh)

convIn = tf.reshape(fc_3, shape=[-1, 8, 16, 2])
# Convolutional Layer #1, 2nd overall layer
conv1 = tf.layers.conv2d(
    inputs=convIn,
    filters=8,
    kernel_size=3,
    padding="same",
    activation=tf.nn.relu)

drop1 = tf.layers.dropout(
    conv1,
    rate=0.5,
    noise_shape=None,
    seed=None,
    training=mode,
    name=None)

conv2 = tf.layers.conv2d(
    inputs=drop1,
    filters=16,
    kernel_size=5,
    padding="same",
    activation=tf.nn.relu)

drop2 = tf.layers.dropout(
    conv2,
    rate=0.5,
    noise_shape=None,
    seed=None,
    training=mode,
    name=None)

conv3 = tf.layers.conv2d(
    inputs=drop2,
    filters=16,
    kernel_size=7,
    padding="same",
    activation=tf.nn.relu)

drop3 = tf.layers.dropout(
    conv3,
    rate=0.5,
    noise_shape=None,
    seed=None,
    training=mode,
    name=None)

conv4 = tf.layers.conv2d(
    inputs=drop3,
    filters=16,
    kernel_size=5,
    padding="same",
    activation=tf.nn.relu)

drop4 = tf.layers.dropout(
    conv4,
    rate=0.5,
    noise_shape=None,
    seed=None,
    training=mode,
    name=None)

conv5 = tf.layers.conv2d(
    inputs=drop4,
    filters=8,
    kernel_size=3,
    padding="same",
    activation=tf.nn.relu)

drop5 = tf.layers.dropout(
    conv5,
    rate=0.5,
    noise_shape=None,
    seed=None,
    training=mode,
    name=None)

convOut = tf.layers.conv2d(
    inputs=drop5,
    filters=2,
    kernel_size=3,
    padding="same",
    activation=None)

y_pred = tf.reshape(convOut,shape=[-1,256])
# y_pred = fc_3

cost = tf.nn.l2_loss((y - y_pred)) / batchSize
# opt = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)
# opt = tf.train.AdagradOptimizer(learningRate,0.5).minimize(cost)
opt = tf.train.AdamOptimizer(learningRate).minimize(cost)
# now we can start training...

# read input  and training data
position_y, training_data = readTrainingData.loadData(path_to_data)
# training_data = np.reshape(training_data, newshape=[-1, 8, 16, 2])
scaling = np.ndarray.max(abs(training_data))
training_data = training_data / scaling
loadNum = len(position_y)
validationNum = round(loadNum * validationProportion)
validationInd = random.sample(range(0, loadNum), validationNum)
trainingInd = [ind for ind in range(0, loadNum) if ind not in validationInd]
validationData = training_data[validationInd]
validationInput = position_y[validationInd]
training_data = training_data[trainingInd]
trainingInput = position_y[trainingInd]
trainingSize = len(training_data)

print("Starting training...")
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for epoch in range(trainingEpochs):
    # c = (epoch * batchSize) % training_data.shape[0]
    batch_training_out = []
    batch_training_in = []
    for currNo in range(0, batchSize):
        r = random.randint(0, trainingSize - 1)
        batch_training_out.append(training_data[r, :])
        batch_training_in.append(trainingInput[r])
    # batch_xs, batch_ys = training_data[c:c+batchSize,:], training_data[c:c+batchSize,:]

    _, currentCost = sess.run([opt, cost], feed_dict={x: batch_training_in, y: batch_training_out})
    print("Epoch %d/%d: cost %f " % (epoch + 1, trainingEpochs, currentCost))
    error.append(currentCost)

    # #test convergence
    # if epoch % 3000 == 0 and epoch != 0:


    if epoch == trainingEpochs - 1:
        plt.figure()
        plt.plot(error)
        plt.ylabel("training cost")
        plt.xlabel('iteration')
        plt.show()
        mode = False
        [valiCost, vout] = sess.run([cost, y_pred], feed_dict={x: validationInput, y: validationData})
        valiCost = valiCost * batchSize / validationNum
        print(" Validation: cost %f " % (valiCost))

        # for i in range(validationNum):
        valiData = readTrainingData.transformToImage(validationData[0, :], [8, 16, 2])
        vout_img = readTrainingData.transformToImage(vout[0, :], [8, 16, 2])
        # scipy.misc.toimage(valiData[:,:,0], cmin=0.0, cmax=1.0).save("inx_%d.png" % i)
        # scipy.misc.toimage(valiData[:, :, 1], cmin=0.0, cmax=1.0).save("iny_%d.png" % i)
        # scipy.misc.toimage(vout_img[:, :, 0], cmin=0.0, cmax=1.0).save("outx_%d.png" % i)
        # scipy.misc.toimage(vout_img[:, :, 1], cmin=0.0, cmax=1.0).save("outy_%d.png" % i)
        print("Y position:", validationInput[0])
        flow.plot_flow_triple(valiData, vout_img)

print("Done")
