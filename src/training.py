import tensorflow as tf
import numpy as np

import models
import random

class MinibatchSampler:
    def __init__(self, trainingData):
        random.shuffle(trainingData)
        self.trainingData = trainingData
        self.trainingDataSize = len(trainingData)
        self.nextIndex = 0
        self.numEpochs = 0
        self.numTotalSamples = 0

    def resetCounters(self):
        self.nextIndex = 0
        self.numEpochs = 0
        self.numTotalSamples = 0

    def nextMinibatch(self, minibatchSize = 4):
        endIndex = self.nextIndex + minibatchSize
        minibatch = []
        while endIndex > self.trainingDataSize:
            minibatch.extend(self.trainingData[self.nextIndex:])
            endIndex -= self.trainingDataSize
            self.nextIndex = 0
            self.numEpochs += 1
        minibatch.extend(self.trainingData[self.nextIndex:endIndex])
        self.numTotalSamples += minibatchSize
        return minibatch

    def getNumEpochs(self):
        return self.numEpochs

    def getNumTotalSamples(self):
        return self.numTotalSamples


def trainParametricNetwork(modelFunc, lossFunc, sampler, minibatchSize=4):
    x = tf.placeholder(tf.float32, shape=[None, 1])
    y = tf.placeholder(tf.float32, shape=[None, 16, 8, 2])
    flagField = tf.placeholder(tf.float32, shape=[None, 16, 8])
    yPred = modelFunc(x)
    loss = lossFunc(yPred, y, flagField)

    init = tf.global_variables_initializer()
    opt = tf.train.AdamOptimizer(0.001).minimize(loss)
    sess = tf.Session()
    sess.run(init)

    for i in range(1000):
        mb = sampler.nextMinibatch(minibatchSize)
        xValues = np.array([ex.x for ex in mb])
        yValues = np.array([ex.y for ex in mb])
        ffValues = np.array([ex.flagField for ex in mb])
        optResult, lossResult = sess.run([opt, loss], {x: xValues, y: yValues, flagField: ffValues})
        print("Loss: {}".format(lossResult))





examples = []
trainParametricNetwork(models.simple_model_1, models.simple_loss_1, examples)