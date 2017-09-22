import tensorflow as tf
import utils

def simpleModel1(x):
    #fc1_w = tf.get_variable("fc1_w", initializer=tf.random_normal([1, 256], stddev=0.1))
    #fc1_b = tf.get_variable("fc1_b", initializer=tf.constant(1.0, shape=[256]))
    #fc1_a = tf.add(tf.matmul(x, fc1_w), fc1_b)
    #fc1_a = tf.tanh(fc1_a)
    # fc1_a = tf.dropout(fc1_a, 0.5)
    fc1_a = tf.layers.dense(x, 256, kernel_initializer=tf.random_normal([1, 256], stddev=0.1))  # activation = tf.tanh
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
    numFeatures = 1
    convSize = 2
    scaleFactor = 1
    #layer = tf.contrib.layers.batch_norm(layer, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True,
    #                                             scope="bn0")
    layer = tf.layers.dense(layer, 8, activation=tf.nn.relu)
    layer = tf.contrib.layers.batch_norm(layer, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True,
                                         scope="bn1")
    layer = tf.layers.dense(layer, 32, activation=tf.nn.relu)
    layer = tf.contrib.layers.batch_norm(layer, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True,
                                                 scope="bn2")
    #layer = tf.nn.dropout(layer, 0.7)
    layer = tf.layers.dense(layer, 2048 * scaleFactor * scaleFactor * numFeatures, activation=tf.nn.relu)
    layer = tf.reshape(layer, [-1, 64 * scaleFactor, 32 * scaleFactor, numFeatures])
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

def simpleLoss1(yPred, y, flagField):
    # loss = tf.reduce_mean(tf.square(yPred - y))
    loss = tf.reduce_mean(tf.abs(yPred - y))
    return loss

def simpleLoss2(yPred, y, flagField):
    loss = tf.reduce_mean(tf.abs(tf.expand_dims(flagField, -1) * (yPred - y)))
    return loss

class NeuralNetwork(object):
    def __init__(self, x, y, yPred, loss):
        self.x = x
        self.y = y
        self.yPred = yPred
        self.loss = loss
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
    def __init__(self, x, y, yPred, loss, flagField):
        super(FlagFieldNN, self).__init__(x, y, yPred, loss)
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

def computeSimpleNN(modelFunc, lossFunc, inputDim = 1, scale=0.25):
    x = tf.placeholder(tf.float32, shape=[None, inputDim])
    y = tf.placeholder(tf.float32, shape=[None, int(64 * scale), int(32 * scale), 2])
    yPred = modelFunc(x)
    flagField = tf.placeholder(tf.float32, shape=[None, int(64 * scale), int(32 * scale)])
    loss = lossFunc(yPred, y, flagField)
    return FlagFieldNN(x, y, yPred, loss, flagField)

def computeSimpleNNWithReg(modelFunc, lossFunc, inputDim = 1):
    x = tf.placeholder(tf.float32, shape=[None, inputDim])
    y = tf.placeholder(tf.float32, shape=[None, 16, 8, 2])
    yPred, regLoss = modelFunc(x)
    flagField = tf.placeholder(tf.float32, shape=[None, 16, 8])
    loss = regLoss + lossFunc(yPred, y, flagField)
    return FlagFieldNN(x, y, yPred, loss, flagField)
