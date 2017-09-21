import tensorflow as tf

import models

def trainParametricNetwork(modelFunc, lossFunc, examples):
    x = tf.placeholder(tf.float32, shape=[None, 1])
    yPred = modelFunc(x)
    loss = lossFunc(yPred, examples.y, examples.flagField)

    init = tf.global_variables_initializer()
    opt = tf.train.AdamOptimizer(0.001).minimize(loss)
    sess = tf.Session()
    sess.run(init)
    


examples = []
trainParametricNetwork(models.simple_model_1, models.simple_loss_1, examples)