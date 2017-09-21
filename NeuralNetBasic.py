import tensorflow as tf
import numpy as np
import scipy.misc

# script for first basic neural net where input layer
# (y coordinate of obstacle) is directly forwarded to output layer with 256 flow values

trainingEpochs = 20
batchSize = 1
inSize = 1 # warning - hard coded to scalar values 1

# set up the network

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32, shape=[None, 8, 16, 2])

xIn = tf.reshape(x, shape=[-1, inSize])  # flatten
fc_1w = tf.Variable(tf.random_normal([inSize, 256], stddev=0.01))
fc_1b = tf.Variable(tf.random_normal([256], stddev=0.01))

fc1 = tf.add(tf.matmul(xIn, fc_1w), fc_1b)
fc1 = tf.nn.tanh(fc1)
# fc1 = tf.nn.dropout(fc1, 0.5)  # plenty of dropout...

# fc_2w = tf.Variable(tf.random_normal([50, inSize], stddev=0.01))  # back to input size
# fc_2b = tf.Variable(tf.random_normal([inSize], stddev=0.01))

# y_pred = tf.add(tf.matmul(fc1, fc_2w), fc_2b)
# y_pred = tf.reshape(y_pred, shape=[-1, 64, 64, 1])
y_pred = fc1
y_pred = tf.reshape(y_pred, shape=[-1, 8, 16, 2])

cost = tf.nn.l2_loss(y - y_pred)
opt = tf.train.GradientDescentOptimizer(0.2).minimize(cost)

# now we can start training...

# read input  and training data
position_y = 0.5
training_data = np.linspace(-1, 4, 256, dtype=float)
training_data = np.reshape(training_data, newshape = [-1, 8,16, 2])

print("Starting training...")
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for epoch in range(trainingEpochs):
    # c = (epoch * batchSize) % densities.shape[0]
    # batch = []
    # for currNo in range(0, batchSize):
    #     r = random.randint(0, loadNum - 1)
    #     batch.append(densities[r])
    # batch_xs, batch_ys = densities[c:c+batchSize,:], densities[c:c+batchSize,:]
    _, currentCost = sess.run([opt, cost], feed_dict={x: position_y, y: training_data})
    print("Epoch %d/%d: cost %f " % (epoch, trainingEpochs, currentCost) )

    if epoch == trainingEpochs - 1:
        [valiCost, vout] = sess.run([cost, y_pred], feed_dict={x: position_y, y: training_data})
        print(" Validation: cost %f , validation cost %f " % (currentCost, valiCost))
        # print("Validation cost %f " % (valiCost) )

        # if epoch == trainingEpochs - 1:
        #     for i in range(len(training_data)):
        #         scipy.misc.toimage(training_data, cmin=0.0, cmax=1.0).save("in_%d.png" % i)
        #         scipy.misc.toimage(np.reshape(vout[i], [8, 16, 2]), cmin=0.0, cmax=1.0).save("out_%d.png" % i)

print("Done")