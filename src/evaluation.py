import tensorflow as tf
import numpy as np
import scipy.ndimage
import copy
import random

class ParametricSimulationExample(object):
    def __init__(self, sim1Result, slice=[1], scale=0.25):
        self.x = [sim1Result.obstacle_pos[i] for i in slice]
        # print(sim1Result.npVel.shape)
        arr = sim1Result.npVel
        arr = np.delete(arr, 2, 2)
        #if(arr.shape[0] < arr.shape[1]):
        #    arr = np.transpose(arr, (1, 0, 2))
        arr = scipy.ndimage.zoom(arr, [scale, scale, 1])
        # print(arr.shape)
        self.y = arr
        func = lambda x: 1.0 if x > 0.0 else 0.0

        obs = sim1Result.obstacles
        #if (obs.shape[0] < obs.shape[1]):
        #    obs = np.transpose(obs)
        obs = np.vectorize(func)(obs)
        self.flagField = scipy.ndimage.zoom(obs, [scale, scale])

class FlagFieldSimulationExample(object):
    def __init__(self, sim1Result, slice=[1], scale=0.25):
        # print(sim1Result.npVel.shape)
        arr = sim1Result.npVel
        arr = np.delete(arr, 2, 2)
        # if(arr.shape[0] < arr.shape[1]):
        #    arr = np.transpose(arr, (1, 0, 2))
        arr = scipy.ndimage.zoom(arr, [scale, scale, 1])
        # print(arr.shape)
        self.y = arr
        func = lambda x: 1.0 if x > 0.0 else 0.0

        obs = sim1Result.obstacles
        # if (obs.shape[0] < obs.shape[1]):
        #    obs = np.transpose(obs)
        obs = np.vectorize(func)(obs)
        self.flagField = scipy.ndimage.zoom(obs, [scale, scale])
        # self.x = scipy.ndimage.zoom(obs, [scale, scale])
        self.x = self.flagField

class DataPartition(object):
    def __init__(self, dataSize, trainingFraction=0.6, validationFraction=0.2):
        self.dataSize = dataSize
        self.trainingFraction = trainingFraction
        self.validationFraction = validationFraction
        indices = range(dataSize)
        random.shuffle(indices)
        trainingEnd = int(dataSize * trainingFraction)
        validationEnd = int(dataSize * (trainingFraction + validationFraction))
        self.trainingIndices = indices[0:trainingEnd]
        self.validationIndices = indices[trainingEnd:validationEnd]
        self.testIndices = indices[validationEnd:dataSize]

    def computeData(self, data, exampleType=ParametricSimulationExample, slice=[1], scale=0.25):
        if len(data) != self.dataSize:
            raise ValueError("DataPartition.computeData(): incompatible data length")

        trainingData = [exampleType(data[i], slice, scale) for i in self.trainingIndices]
        validationData = [exampleType(data[i], slice, scale) for i in self.validationIndices]
        testData = [exampleType(data[i], slice, scale) for i in self.testIndices]

        return trainingData, validationData, testData


def getFeedDict(network, data):
    xValues = np.array([ex.x for ex in data])
    yValues = np.array([ex.y for ex in data])
    ffValues = np.array([ex.flagField for ex in data])
    #print(xValues.shape)
    #print(yValues.shape)
    #print(ffValues.shape)
    #print("shapes: ")
    #for ex in data:
    #    print(ex.y.shape)
    #    print(ex.flagField.shape)
    #    print("")
    return {network.x: xValues, network.y: yValues, network.flagField: ffValues}

def validateModel(flagFieldNN, validationData, name="final"):
    sess = tf.Session()
    flagFieldNN.load(sess, name)
    yPred, lossResult = sess.run([flagFieldNN.yPred, flagFieldNN.loss], getFeedDict(flagFieldNN, validationData))
    print("Validation loss: {}".format(lossResult))
