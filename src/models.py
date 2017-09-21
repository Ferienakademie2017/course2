import tensorflow as tf

def simpleModel1(x):
    #fc1_w = tf.get_variable("fc1_w", initializer=tf.random_normal([1, 256], stddev=0.1))
    #fc1_b = tf.get_variable("fc1_b", initializer=tf.constant(1.0, shape=[256]))
    #fc1_z = tf.add(tf.matmul(x, fc1_w), fc1_b)
    #fc1_a = tf.tanh(fc1_z)
    # fc1_a = tf.dropout(fc1_a, 0.5)
    fc1_a = tf.layers.dense(x, 256)  # activation = tf.tanh
    y_pred = tf.reshape(fc1_a, [-1, 16, 8, 2])
    return y_pred

def simpleLoss1(yPred, y, flagField):
    loss = tf.reduce_mean(tf.abs(yPred - y))
    return loss

class NeuralNetwork(object):
    def __init__(self, x, y, yPred, loss):
        self.x = x
        self.y = y
        self.yPred = yPred
        self.loss = loss

    def compute(self, x, sess):
        return sess.run(self.yPred, {self.x: x})

class FlagFieldNN(NeuralNetwork):
    def __init__(self, x, y, yPred, loss, flagField):
        super(FlagFieldNN, self).__init__(x, y, yPred, loss)
        self.flagField = flagField


def computeNN1():
    x = tf.placeholder(tf.float32, shape=[None, 1])
    y = tf.placeholder(tf.float32, shape=[None, 16, 8, 2])
    yPred = simpleModel1(x)
    flagField = tf.placeholder(tf.float32, shape=[None, 16, 8])
    loss = simpleLoss1(yPred, y, flagField)
    return FlagFieldNN(x, y, yPred, loss, flagField)

