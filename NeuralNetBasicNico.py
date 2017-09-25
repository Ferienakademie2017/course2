import tensorflow as tf
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

# path_to_data = r'C:\Users\Annika\Saved Games\Desktop\course2\trainingData\trainingKarman32.p'
# path_to_data = r'C:\Users\Nico\Documents\Ferienakademie\course2\trainingData\trainingKarman32i100.p'
# path_to_data = r'C:\Users\Nico\Documents\Ferienakademie\course2\trainingData\trainingKarman32.p'
path_to_data = r'C:\Users\Nico\Documents\Ferienakademie\course2\trainingData\trainingKarman32_1000randu.p'
trainingEpochs = 5000
batchSize = 128
inSize = 1  # warning - hard coded to scalar values 1
validationProportion = 0.05
learningRate = 0.5
error = []

# set up the network

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)  # training data

xIn = tf.reshape(x, shape=[-1, inSize])  # flatten
size1 = 64
fc_1w = tf.Variable(tf.random_normal([inSize, size1], stddev=0.01))
fc_1b = tf.Variable(tf.random_normal([size1], stddev=0.01))

fc_1 = tf.add(tf.matmul(xIn, fc_1w), fc_1b)
# fc_1 = tf.nn.tanh(fc_1)
fc_1 = tf.nn.relu(fc_1)
# fc_1 = tf.nn.dropout(fc_1, 0.5)  # plenty of dropout...

size2 = 128
fc_2w = tf.Variable(tf.random_normal([size1, size2], stddev=0.01))  # back to input size
fc_2b = tf.Variable(tf.random_normal([size2], stddev=0.01))

fc_2 = tf.add(tf.matmul(fc_1, fc_2w), fc_2b)
# fc_2 = tf.nn.tanh(fc_2)
fc_2 = tf.nn.relu(fc_2)
# fc_2 = tf.nn.dropout(fc_2, 0.5)  # plenty of dropout...

size3 = 256
fc_3w = tf.Variable(tf.random_normal([size2, size3], stddev=0.01))  # back to input size
fc_3b = tf.Variable(tf.random_normal([size3], stddev=0.01))

fc_3 = tf.add(tf.matmul(fc_2, fc_3w), fc_3b)
fc_3 = tf.nn.tanh(fc_3)
# fc_3 = tf.nn.dropout(fc_3, 0.5)  # plenty of dropout...

# y_pred = tf.add(tf.matmul(fc1, fc_2w), fc_2b)
# y_pred = tf.reshape(y_pred, shape=[-1, 64, 64, 1])
y_pred = fc_3
# y_pred = tf.reshape(y_pred, shape=[-1, 8, 16, 2])

cost = tf.nn.l2_loss((y - y_pred)) / batchSize
# opt = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)
opt = tf.train.AdagradOptimizer(learningRate,0.5).minimize(cost)

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
