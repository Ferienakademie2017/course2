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


def trainNetwork(flagFieldNN, sampler, minibatchSize=4, numMinibatches=200):
    init = tf.global_variables_initializer()
    opt = tf.train.AdamOptimizer(0.001).minimize(flagFieldNN.loss)
    sess = tf.Session()
    sess.run(init)

    for i in range(numMinibatches):
        mb = sampler.nextMinibatch(minibatchSize)
        xValues = np.array([ex.x for ex in mb])
        yValues = np.array([ex.y for ex in mb])
        ffValues = np.array([ex.flagField for ex in mb])
        optResult, lossResult = sess.run([opt, flagFieldNN.loss],
                                         {flagFieldNN.x: xValues,
                                          flagFieldNN.y: yValues,
                                          flagFieldNN.flagField: ffValues})
        print("Loss: {}".format(lossResult))
        # todo: evtl. hier eine ErrorReporter-Klasse rein
        # todo: oder gleich Klasse, die auch noch die Abbruchbedingung festlegt oder eine Änderung der Learning Rate

class SimulationExample(object):
    def __init__(self, sim1Result):
        self.x = sim1Result.obstacle_pos
        self.y = sim1Result.npVel
        func = lambda x: 1.0 if x < 0.0 else 0.0
        self.flagField = np.vectorize(func)(sim1Result.obstacles)

def generateParametricExamples(data, trainingFraction=0.6, validationFraction=0.2):
    dataSize = len(data)
    trainingEnd = int(dataSize*trainingFraction)
    validationEnd = int(dataSize*(trainingFraction+validationFraction))
    trainingData = [SimulationExample(res) for res in data[0:trainingEnd]]
    validationData = [SimulationExample(res) for res in data[trainingEnd:validationEnd]]
    testData = [SimulationExample(res) for res in data[validationEnd:dataSize]]
    return trainingData, validationData, testData



examples = []
trainNetwork(models.computeNN1(), MinibatchSampler(examples))