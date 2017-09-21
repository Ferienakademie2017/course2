# coding: utf-8
import tensorflow as tf
import numpy as np

import random
import evaluation

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


def trainNetwork(flagFieldNN, sampler, lossLogger, minibatchSize=4, numMinibatches=200):
    init = tf.global_variables_initializer()
    opt = tf.train.AdamOptimizer(0.001).minimize(flagFieldNN.loss)
    sess = tf.Session()
    sess.run(init)

    for i in range(numMinibatches):
        mb = sampler.nextMinibatch(minibatchSize)
        optResult, lossResult = sess.run([opt, flagFieldNN.loss], evaluation.getFeedDict(flagFieldNN, mb))
        lossLogger.logLoss(i, lossResult)
        # todo: evtl. hier eine ErrorReporter-Klasse rein
        # todo: oder gleich Klasse, die auch noch die Abbruchbedingung festlegt oder eine Änderung der Learning Rate


#examples = []
#trainNetwork(models.computeNN1(), MinibatchSampler(examples))
