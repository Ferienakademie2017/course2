import tensorflow as tf
import utils
import numpy as np
import layers

def simpleModel1(x):
    fc1_w = tf.get_variable("fc1_w", initializer=tf.random_normal([1, 256], stddev=0.1))
    fc1_b = tf.get_variable("fc1_b", initializer=tf.constant(1.0, shape=[256]))
    fc1_a = tf.add(tf.matmul(x, fc1_w), fc1_b)
    #fc1_a = tf.tanh(fc1_a)
    #fc1_a = tf.layers.dropout(fc1_a, 0.5)
    #fc1_a = tf.layers.dense(x, 256, kernel_initializer=tf.random_normal([1, 256], stddev=0.1))  # activation = tf.tanh
    y_pred = tf.reshape(fc1_a, [-1, 16, 8, 2])
    return y_pred

def simpleModel2(x):
    layer = tf.layers.dense(x, 2, activation=tf.nn.relu)
    layer = tf.layers.dense(layer, 4, activation=tf.nn.relu)
    layer = tf.layers.dense(layer, 8, activation=tf.nn.relu)
    layer = tf.layers.dense(layer, 16, activation=tf.nn.relu)
    layer = tf.layers.dense(layer, 32, activation=tf.nn.relu)
    layer = tf.layers.dense(layer, 64, activation=tf.nn.relu)
    layer = tf.layers.dense(layer, 128, activation=tf.nn.relu)
    fc2_a = tf.layers.dense(layer, 256)
    y_pred = tf.reshape(fc2_a, [-1, 16, 8, 2])
    return y_pred

def simpleModel3(x):
    layer = tf.layers.dense(x, 64, activation=tf.nn.relu)
    layer = tf.reshape(layer, [-1, 8, 4, 2])
    layer = tf.contrib.layers.conv2d_transpose(layer, 8, [3, 3], [2, 2], "SAME", activation_fn=tf.nn.relu,
                                               weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                               biases_initializer=tf.constant_initializer(0.0))
    layer = tf.contrib.layers.conv2d(layer, 2, [3, 3], [1, 1], "SAME", activation_fn=None,
                                     weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                     biases_initializer=tf.constant_initializer(0.0))
    #y_pred = tf.reshape(layer, [-1, 16, 8, 2])
    return layer

def simpleModel4(x):
    regFactor = 1.0
    weights = tf.get_variable("fc1_w", initializer=tf.random_normal([2, 8], stddev=0.1))
    biases = tf.get_variable("fc1_b", initializer=tf.constant(1.0, shape=[8]))
    layer = tf.add(tf.matmul(x, weights), biases)
    loss = tf.reduce_sum(tf.square(weights))
    #layer = tf.layers.dense(x, 8)
    #layer = tf.nn.dropout(layer, 0.7)
    layer = tf.nn.relu(layer)
    #layer = tf.layers.dense(x, 8, activation=tf.nn.relu)
    # layer = tf.layers.dense(x, 16, activation=tf.nn.relu)
    #layer = tf.layers.dense(layer, 256)
    weights2 = tf.get_variable("fc2_w", initializer=tf.random_normal([8, 256], stddev=0.1))
    biases2 = tf.get_variable("fc2_b", initializer=tf.constant(1.0, shape=[256]))
    layer = tf.add(tf.matmul(layer, weights2), biases2)
    layer = tf.reshape(layer, [-1, 16, 8, 2])
    loss = loss + tf.reduce_sum(tf.square(weights2))
    #layer = tf.contrib.layers.conv2d(layer, 2, [3, 3], [1, 1], "SAME", activation_fn=None,
    #                                 weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
    #                                 biases_initializer=tf.constant_initializer(0.0))
    return layer, regFactor * loss

def simpleModel5(x):
    layer = x
    numFeatures = 1
    convSize = 2
    scaleFactor = 2
    #layer = tf.contrib.layers.batch_norm(layer, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True,
    #                                             scope="bn0")
    layer = tf.layers.dense(layer, 8, activation=tf.nn.relu)
    layer = tf.contrib.layers.batch_norm(layer, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True,
                                         scope="bn1")
    layer = tf.layers.dense(layer, 32, activation=tf.nn.relu)
    layer = tf.contrib.layers.batch_norm(layer, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True,
                                                 scope="bn2")
    #layer = tf.nn.dropout(layer, 0.7)
    layer = tf.layers.dense(layer, 128 * scaleFactor * scaleFactor * numFeatures, activation=tf.nn.relu)
    layer = tf.reshape(layer, [-1, 16 * scaleFactor, 8 * scaleFactor, numFeatures])
    layer = tf.contrib.layers.batch_norm(layer, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True,
                                                 scope="bn3")
    for i in range(5):
        oldLayer = layer
        layer = tf.contrib.layers.conv2d(layer, numFeatures, [convSize, convSize], [1, 1], "SAME", activation_fn=None,
                                         weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                         biases_initializer=tf.constant_initializer(0.0))
        layer = tf.contrib.layers.batch_norm(layer, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True,
                                     scope="batch_norm1_{}".format(i))
        layer = tf.nn.relu(layer)
        layer = tf.contrib.layers.conv2d(layer, numFeatures, [convSize, convSize], [1, 1], "SAME", activation_fn=None,
                                         weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                         biases_initializer=tf.constant_initializer(0.0))
        layer = tf.contrib.layers.batch_norm(layer, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True,
                                         scope="batch_norm2_{}".format(i))
        # layer = tf.nn.relu(layer)
        layer = layer + oldLayer
        # layer = tf.nn.dropout(layer, 0.8)

    layer = tf.contrib.layers.conv2d(layer, 2, [convSize, convSize], [scaleFactor, scaleFactor], "SAME", activation_fn=None,
                                     weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                     biases_initializer=tf.constant_initializer(0.0))

    return layer

def simpleModel6(x):
    layer = x
    numFeatures = 2
    convSize = 2
    scaleFactor = 1
    #layer = tf.contrib.layers.batch_norm(layer, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True,
    #                                             scope="bn0")
    #layer = tf.layers.dense(layer, 8, activation=tf.nn.relu)
    #layer = tf.contrib.layers.batch_norm(layer, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True,
    #                                     scope="bn1")
    #layer = tf.layers.dense(layer, 32, activation=tf.nn.relu)
    #layer = tf.contrib.layers.batch_norm(layer, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True,
    #                                             scope="bn2")
    #layer = tf.nn.dropout(layer, 0.7)
    layer = tf.layers.dense(layer, 2048 * scaleFactor * scaleFactor * numFeatures, activation=tf.nn.relu)
    layer = tf.reshape(layer, [-1, 64 * scaleFactor, 32 * scaleFactor, numFeatures])
    layer = tf.contrib.layers.batch_norm(layer, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True,
                                                 scope="bn3")
    for i in range(1):
        oldLayer = layer
        layer = tf.contrib.layers.conv2d(layer, numFeatures, [convSize, convSize], [1, 1], "SAME", activation_fn=None,
                                         weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                         biases_initializer=tf.constant_initializer(0.0))
        layer = tf.contrib.layers.batch_norm(layer, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True,
                                     scope="batch_norm1_{}".format(i))
        layer = tf.nn.relu(layer)
        layer = tf.contrib.layers.conv2d(layer, numFeatures, [convSize, convSize], [1, 1], "SAME", activation_fn=None,
                                         weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                         biases_initializer=tf.constant_initializer(0.0))
        layer = tf.contrib.layers.batch_norm(layer, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True,
                                         scope="batch_norm2_{}".format(i))
        # layer = tf.nn.relu(layer)
        layer = layer + oldLayer
        # layer = tf.nn.dropout(layer, 0.8)

    layer = tf.contrib.layers.conv2d(layer, 2, [convSize, convSize], [scaleFactor, scaleFactor], "SAME", activation_fn=None,
                                     weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                     biases_initializer=tf.constant_initializer(0.0))

    return layer

def simpleModel7(x):
    layer = x
    layer = tf.layers.dense(layer, 8, activation=tf.tanh)
    layer = tf.contrib.layers.batch_norm(layer, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True,
                                         scope="bn1")
    #layer = tf.layers.dense(layer, 32, activation=tf.nn.relu)
    #layer = tf.layers.dense(layer, 32, activation=tf.nn.relu)
    #layer = tf.layers.dense(layer, 32, activation=tf.nn.relu)
    layer = tf.layers.dense(layer, 32, activation=tf.tanh)
    layer = tf.contrib.layers.batch_norm(layer, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True,
                                         scope="bn2")
    #layer = tf.layers.dense(layer, 32, activation=tf.nn.relu)
    #layer = tf.contrib.layers.batch_norm(layer, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True,
    #                                     scope="bn21")
    #layer = tf.layers.dense(layer, 32, activation=tf.nn.relu)
    #layer = tf.contrib.layers.batch_norm(layer, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True,
    #                                     scope="bn22")
    #layer = tf.layers.dense(layer, 32, activation=tf.nn.relu)
    #layer = tf.contrib.layers.batch_norm(layer, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True,
    #                                     scope="bn23")
    layer = tf.layers.dense(layer, 256, activation=tf.tanh)
    layer = tf.contrib.layers.batch_norm(layer, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True,
                                         scope="bn3")
    #layer = tf.layers.dense(layer, 4096)
    #layer = tf.contrib.layers.batch_norm(layer, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True,
    #                                     scope="bn4")
    layer = tf.layers.dense(layer, 4096)
    layer = tf.reshape(layer, [-1, 64, 32, 2])

    return layer

def simpleModel8(x):
    layer = x
    numFeatures = 2
    convSize = 5
    scaleFactor = 1
    layer = tf.reshape(layer, [-1, 64, 32, 1])

    layer = tf.contrib.layers.conv2d(layer, numFeatures, [convSize, convSize], [1, 1], "SAME", activation_fn=tf.nn.relu,
                                     weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                     biases_initializer=tf.constant_initializer(0.0))

    for i in range(10):
        oldLayer = layer
        layer = tf.contrib.layers.conv2d(layer, numFeatures, [convSize, convSize], [1, 1], "SAME", activation_fn=None,
                                         weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                         biases_initializer=tf.constant_initializer(0.0))
        layer = tf.contrib.layers.batch_norm(layer, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True,
                                     scope="batch_norm1_{}".format(i))
        layer = tf.nn.relu(layer)
        layer = tf.contrib.layers.conv2d(layer, numFeatures, [convSize, convSize], [1, 1], "SAME", activation_fn=None,
                                         weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                         biases_initializer=tf.constant_initializer(0.0))
        layer = tf.contrib.layers.batch_norm(layer, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True,
                                         scope="batch_norm2_{}".format(i))
        # layer = tf.nn.relu(layer)
        layer = layer + oldLayer
        # layer = tf.nn.dropout(layer, 0.8)

    layer = tf.contrib.layers.conv2d(layer, 2, [convSize, convSize], [scaleFactor, scaleFactor], "SAME", activation_fn=None,
                                     weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                     biases_initializer=tf.constant_initializer(0.0))

    return layer

def simpleModel9(x):
    layer = x
    numFeatures = 10
    convSize = 5
    scaleFactor = 1
    zoomSteps = 3
    zoomLayers = []
    # zoomLayers
    #layer = tf.reshape(layer, [-1, 64, 32, 1])
    layer = tf.expand_dims(layer, -1)


    for i in range(zoomSteps):
        zoomLayers.append(layer)
        layer = tf.contrib.layers.conv2d(layer, numFeatures, [convSize, convSize], [2, 2], "SAME", activation_fn=tf.nn.relu,
                                     weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                     biases_initializer=tf.constant_initializer(0.0))
        layer = tf.contrib.layers.batch_norm(layer, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True,
                                             scope="batch_norm0_{}".format(i))

    for i in range(10):
        oldLayer = layer
        layer = tf.contrib.layers.conv2d(layer, numFeatures, [convSize, convSize], [1, 1], "SAME", activation_fn=None,
                                         weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                         biases_initializer=tf.constant_initializer(0.0))
        layer = tf.contrib.layers.batch_norm(layer, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True,
                                     scope="batch_norm1_{}".format(i))
        layer = tf.nn.relu(layer)
        layer = tf.contrib.layers.conv2d(layer, numFeatures, [convSize, convSize], [1, 1], "SAME", activation_fn=None,
                                         weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                         biases_initializer=tf.constant_initializer(0.0))
        layer = tf.contrib.layers.batch_norm(layer, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True,
                                         scope="batch_norm2_{}".format(i))
        # layer = tf.nn.relu(layer)
        layer = layer + oldLayer
        # layer = tf.nn.dropout(layer, 0.8)

    for i in range(zoomSteps):
        layer = tf.contrib.layers.conv2d_transpose(layer, numFeatures, [convSize, convSize], [2, 2], "SAME",
                                         activation_fn=tf.nn.relu,
                                         weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                         biases_initializer=tf.constant_initializer(0.0))
        layer = tf.contrib.layers.batch_norm(layer, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True,
                                             scope="batch_norm3_{}".format(i))
        layer = tf.concat([layer, zoomLayers[zoomSteps - 1 - i]], 3)

        layer = tf.contrib.layers.conv2d(layer, numFeatures, [convSize, convSize], [1, 1], "SAME",
                                         activation_fn=tf.nn.relu,
                                         weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                         biases_initializer=tf.constant_initializer(0.0))
        layer = tf.contrib.layers.batch_norm(layer, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True,
                                         scope="batch_norm4_{}".format(i))
        layer = tf.contrib.layers.conv2d(layer, numFeatures, [convSize, convSize], [1, 1], "SAME",
                                         activation_fn=tf.nn.relu,
                                         weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                         biases_initializer=tf.constant_initializer(0.0))
        layer = tf.contrib.layers.batch_norm(layer, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True,
                                         scope="batch_norm5_{}".format(i))

    layer = tf.contrib.layers.conv2d(layer, 2, [convSize, convSize], [scaleFactor, scaleFactor], "SAME", activation_fn=None,
                                     weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                     biases_initializer=tf.constant_initializer(0.0))

    return layer

def timeStepModel1(x, phase):
    layer = x
    numFeatures = 4
    convSize = 4
    scaleFactor = 1
    zoomSteps = 1
    act = layers.lrelu # tf.nn.relu # tf.tanh
    zoomLayers = []
    # zoomLayers

    for i in range(zoomSteps):
        zoomLayers.append(layer)
        layer = tf.contrib.layers.conv2d(layer, numFeatures, [convSize, convSize], [2, 2], "SAME",
                                         activation_fn=act,
                                         weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                         biases_initializer=tf.constant_initializer(0.0),scope="conv1_{}".format(i))
        layer = tf.contrib.layers.batch_norm(layer, decay=0.9, is_training=phase, updates_collections=None, epsilon=1e-5, scale=True,
                                             scope="batch_norm0_{}".format(i))

    for i in range(4):
        oldLayer = layer
        layer = tf.contrib.layers.conv2d(layer, numFeatures, [convSize, convSize], [1, 1], "SAME", activation_fn=None,
                                         weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                         biases_initializer=tf.constant_initializer(0.0),scope="resnet1_{}".format(i))
        layer = tf.contrib.layers.batch_norm(layer, decay=0.9, is_training=phase, updates_collections=None, epsilon=1e-5, scale=True,
                                             scope="batch_norm1_{}".format(i))
        layer = act(layer)
        layer = tf.contrib.layers.conv2d(layer, numFeatures, [convSize, convSize], [1, 1], "SAME", activation_fn=None,
                                         weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                         biases_initializer=tf.constant_initializer(0.0),scope="resnet2_{}".format(i))
        layer = tf.contrib.layers.batch_norm(layer, decay=0.9, is_training=phase, updates_collections=None, epsilon=1e-5, scale=True,
                                             scope="batch_norm2_{}".format(i))
        # layer = tf.nn.relu(layer)
        layer = layer + oldLayer
        # layer = tf.nn.dropout(layer, 0.8)

    for i in range(zoomSteps):
        layer = tf.contrib.layers.conv2d_transpose(layer, numFeatures, [convSize, convSize], [2, 2], "SAME",
                                                   activation_fn=act,
                                                   weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                                   biases_initializer=tf.constant_initializer(0.0),scope="tconv1_{}".format(i))
        layer = tf.contrib.layers.batch_norm(layer, decay=0.9, is_training=phase, updates_collections=None, epsilon=1e-5, scale=True,
                                             scope="batch_norm3_{}".format(i))
        layer = tf.concat([layer, zoomLayers[zoomSteps - 1 - i]], 3)

        layer = tf.contrib.layers.conv2d(layer, numFeatures, [convSize, convSize], [1, 1], "SAME",
                                         activation_fn=act,
                                         weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                         biases_initializer=tf.constant_initializer(0.0),scope="tconv2_{}".format(i))
        layer = tf.contrib.layers.batch_norm(layer, decay=0.9, is_training=phase, updates_collections=None, epsilon=1e-5, scale=True,
                                             scope="batch_norm4_{}".format(i))
        layer = tf.contrib.layers.conv2d(layer, numFeatures, [convSize, convSize], [1, 1], "SAME",
                                         activation_fn=act,
                                         weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                         biases_initializer=tf.constant_initializer(0.0),scope="tconv3_{}".format(i))
        layer = tf.contrib.layers.batch_norm(layer, decay=0.9, is_training=phase, updates_collections=None, epsilon=1e-5, scale=True,
                                             scope="batch_norm5_{}".format(i))

    layer = tf.contrib.layers.conv2d(layer, 2, [convSize, convSize], [scaleFactor, scaleFactor], "SAME",
                                     activation_fn=None,
                                     weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                     biases_initializer=tf.constant_initializer(0.0),scope="output")

    return layer

def timeStepModel2(x, phase):
    layer = x
    numFeatures = 3
    convSize = 4
    scaleFactor = 1
    act = layers.lrelu  # tf.nn.relu # tf.tanh

    for i in range(5):
        oldLayer = layer
        layer = tf.contrib.layers.conv2d(layer, numFeatures, [convSize, convSize], [1, 1], "SAME", activation_fn=None,
                                         weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                         biases_initializer=tf.constant_initializer(0.0),scope="resnet1_{}".format(i))
        layer = tf.contrib.layers.batch_norm(layer, decay=0.9, is_training=phase, updates_collections=None, epsilon=1e-5, scale=True,
                                             scope="batch_norm1_{}".format(i))
        layer = act(layer)
        layer = tf.contrib.layers.conv2d(layer, numFeatures, [convSize, convSize], [1, 1], "SAME", activation_fn=None,
                                         weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                         biases_initializer=tf.constant_initializer(0.0),scope="resnet2_{}".format(i))
        layer = tf.contrib.layers.batch_norm(layer, decay=0.9, is_training=phase, updates_collections=None, epsilon=1e-5, scale=True,
                                             scope="batch_norm2_{}".format(i))
        # layer = tf.nn.relu(layer)
        layer = layer + oldLayer
        # layer = tf.nn.dropout(layer, 0.8)

    layer = tf.contrib.layers.conv2d(layer, 2, [convSize, convSize], [scaleFactor, scaleFactor], "SAME",
                                     activation_fn=None,
                                     weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                     biases_initializer=tf.constant_initializer(0.0),scope="output")

    return layer

def rungeKuttaModel(x,phase,timeStepModel1,timeStepModel2):
    layer = timeStepModel1(x,phase)
    tf.expand_dims(layer,-1)
    tf.expand_dims(x,-1)
    layer2 = timeStepModel2(layer,x,phase)

    return layer2

def simpleLoss1(yPred, y, flagField):
    # loss = tf.reduce_mean(tf.square(yPred - y))
    loss = tf.reduce_mean(tf.abs(yPred - y))
    return loss

def simpleLoss2(yPred, y, flagField):
    loss = tf.reduce_mean(tf.abs(tf.expand_dims(flagField, -1) * (yPred - y)))
    # loss = tf.reduce_mean(tf.square(tf.expand_dims(flagField, -1) * (yPred - y)))
    return loss

def simpleLoss3(yPred, y, flagField):
    obs = tf.expand_dims(flagField, -1)
    loss = tf.reduce_mean(tf.abs((yPred - y)))
    # divField = yPred[:, 2:, 1:-1, 0] - yPred[:, :-2, 1:-1, 0] + yPred[:, 1:-1, 2:, 1] - yPred[:, 1:-1, :-2, 1]
    # divField = yPred[:, 1:, :-1, 0] - yPred[:, :-1, :-1, 0] + yPred[:, :-1, 1:, 1] - yPred[:, :-1, :-1, 1]
    # loss += 0.01 * tf.nn.l2_loss(divField * flagField[:, :-1, :-1])
    # loss = tf.reduce_mean(tf.square(tf.expand_dims(flagField, -1) * (yPred - y)))
    return loss

def multiStepLoss(yPred, y, flagField):
    obs = tf.expand_dims(flagField, -1)
    loss = tf.reduce_mean(tf.abs((yPred - y)))
    return loss

def multiStepLoss2(yPred, y, flagField):
    #Exponential Decay
    a = 0.8
    shape_yPred = yPred.get_shape().as_list()
    a_List = []
    for ind in range(shape_yPred[-1]):
        a_List.append(pow(a,ind))
    yPredDecay = tf.tensordot((yPred - y),tf.convert_to_tensor(a_List),[[4],[0]])
    obs = tf.expand_dims(flagField, -1)
    loss = tf.reduce_mean(tf.abs(yPredDecay))
    divy = obs*yPred
    divField = divy[:, 1:, :-1, 0,:] - divy[:, :-1, :-1, 0,:] + divy[:, :-1, 1:, 1,:] - divy[:, :-1, :-1, 1,:]
    loss += 0.0003 * tf.nn.l2_loss(divField)
    return loss

class NeuralNetwork(object):
    def __init__(self, x, y, yPred, loss, phase):
        self.x = x
        self.y = y
        self.yPred = yPred
        self.loss = loss
        self.phase = phase
        self.saver = tf.train.Saver()

    def compute(self, x, sess):
        return sess.run(self.yPred, {self.x: x})

    def save(self, sess, name):
        name = "training/{}.ckpt".format(name)
        utils.ensureDir(name)
        self.saver.save(sess, name)

    def load(self, sess, name):
        name = "training/{}.ckpt".format(name)
        self.saver.restore(sess, name)

class FlagFieldNN(NeuralNetwork):
    def __init__(self, x, y, yPred, loss, phase, flagField):
        super(FlagFieldNN, self).__init__(x, y, yPred, loss, phase)
        self.flagField = flagField


def computeNN1():
    return computeSimpleNN(simpleModel1, simpleLoss1)

def computeNN2():
    return computeSimpleNN(simpleModel2, simpleLoss1)

def computeNN3():
    return computeSimpleNN(simpleModel3, simpleLoss1, inputDim=2)

def computeNN4():
    return computeSimpleNNWithReg(simpleModel4, simpleLoss2, inputDim=2)

def computeNN5():
    return computeSimpleNN(simpleModel5, simpleLoss2, inputDim=2)

def computeNN6():
    return computeSimpleNN(simpleModel6, simpleLoss2, inputDim=2, scale=1)

def computeNN7():
    return computeSimpleNN(simpleModel7, simpleLoss2, inputDim=2, scale=1)

def computeNN8():
    return computeConvNN(simpleModel8, simpleLoss2, scale=1)

def computeNN9():
    return computeConvNN(simpleModel9, simpleLoss3, scale=1)

def computeTimeStepNN1():
    return computeTimeStepNN(timeStepModel1, simpleLoss3, scale=1)

def computeMultipleTimeStepNN1(numTimeSteps):
    return computeMultipleTimeStepNN(timeStepModel2, multiStepLoss, scale=1,numTimeSteps=numTimeSteps)

def computeMultipleTimeStepNN2(numTimeSteps):
    return computeMultipleTimeStepNN(timeStepModel2, multiStepLoss2, scale=1,numTimeSteps=numTimeSteps)

def computeMultipleTimeStepNN3(numTimeSteps):
    return computeMultipleTimeStepNN(timeStepModel1, multiStepLoss2, scale=1,numTimeSteps=numTimeSteps)

def computeSimpleNN(modelFunc, lossFunc, inputDim = 1, scale=0.25):
    phase = tf.placeholder(tf.bool, name='phase')
    x = tf.placeholder(tf.float32, shape=[None, inputDim])
    y = tf.placeholder(tf.float32, shape=[None, int(64 * scale), int(32 * scale), 2])
    yPred = modelFunc(x)
    flagField = tf.placeholder(tf.float32, shape=[None, int(64 * scale), int(32 * scale)])
    loss = lossFunc(yPred, y, flagField)
    return FlagFieldNN(x, y, yPred, loss, phase, flagField)

def computeConvNN(modelFunc, lossFunc, scale=0.25):
    phase = tf.placeholder(tf.bool, name='phase')
    x = tf.placeholder(tf.float32, shape=[None, int(64 * scale), int(32 * scale)])
    y = tf.placeholder(tf.float32, shape=[None, int(64 * scale), int(32 * scale), 2])
    yPred = modelFunc(x)
    flagField = tf.placeholder(tf.float32, shape=[None, int(64 * scale), int(32 * scale)])
    loss = lossFunc(yPred, y, flagField)
    return FlagFieldNN(x, y, yPred, loss, phase, flagField)

def computeTimeStepNN(modelFunc, lossFunc):
    phase = tf.placeholder(tf.bool, name='phase')
    x = tf.placeholder(tf.float32, shape=[None, None, None, 3])
    y = tf.placeholder(tf.float32, shape=[None, None, None, 2])
    yPred = modelFunc(x, phase)
    flagField = tf.placeholder(tf.float32, shape=[None, None, None])
    loss = lossFunc(yPred, y, flagField)
    return FlagFieldNN(x, y, yPred, loss, phase, flagField)

def computeSimpleNNWithReg(modelFunc, lossFunc, inputDim = 1):
    phase = tf.placeholder(tf.bool, name='phase')
    x = tf.placeholder(tf.float32, shape=[None, inputDim])
    y = tf.placeholder(tf.float32, shape=[None, 16, 8, 2])
    yPred, regLoss = modelFunc(x)
    flagField = tf.placeholder(tf.float32, shape=[None, 16, 8])
    loss = regLoss + lossFunc(yPred, y, flagField)
    return FlagFieldNN(x, y, yPred, loss, phase, flagField)

def computeMultipleTimeStepNN(modelFunc, lossFunc, scale=0.25,numTimeSteps = 1):
    phase = tf.placeholder(tf.bool, name='phase')
    x = tf.placeholder(tf.float32, shape=[None, int(64 * scale), int(32 * scale), 3])
    y = tf.placeholder(tf.float32, shape=[None, int(64 * scale), int(32 * scale), 2,numTimeSteps])
    network_List = []
    #y_List = []
    with tf.variable_scope("MultiStep") as scope:
        network_List.append(modelFunc(x, phase))
        #y_List.append(y)
        for ind in range(numTimeSteps-1):
            scope.reuse_variables()
            network_List.append(modelFunc(tf.concat((network_List[ind],x[:,:,:,-1:]),-1),phase))
            #y_List.append(tf.placeholder(tf.float32, shape=[None, int(64 * scale), int(32 * scale), 2]))

        yPreds = tf.concat([tf.expand_dims(network,-1) for network in network_List],-1)

        flagField = tf.placeholder(tf.float32, shape=[None, int(64 * scale), int(32 * scale)])
        loss = lossFunc(yPreds, y, tf.expand_dims(flagField,-1))
    return FlagFieldNN(x, y, yPreds, loss, phase, flagField)