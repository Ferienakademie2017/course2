import tensorflow as tf
import numpy as np
import scipy.ndimage
import copy

class ParametricSimulationExample(object):
    def __init__(self, sim1Result, slice=[1]):
        self.x = [sim1Result.obstacle_pos[i] for i in slice]
        # print(sim1Result.npVel.shape)
        arr = sim1Result.npVel
        arr = np.delete(arr, 2, 2)
        #if(arr.shape[0] < arr.shape[1]):
        #    arr = np.transpose(arr, (1, 0, 2))
        arr = scipy.ndimage.zoom(arr, [0.25, 0.25, 1])
        # print(arr.shape)
        self.y = arr
        func = lambda x: 1.0 if x > 0.0 else 0.0

        obs = sim1Result.obstacles
        #if (obs.shape[0] < obs.shape[1]):
        #    obs = np.transpose(obs)
        obs = scipy.ndimage.zoom(obs, [0.25, 0.25])
        self.flagField = np.vectorize(func)(obs)

def generateParametricExamples(data, trainingFraction=0.6, validationFraction=0.2, exampleType=ParametricSimulationExample, slice=[1]):
    data = [ex for ex in data if ex.npVel.shape[0] > ex.npVel.shape[1]]
    dataSize = len(data)
    trainingEnd = int(dataSize*trainingFraction)
    validationEnd = int(dataSize*(trainingFraction+validationFraction))
    trainingData = [exampleType(res, slice) for res in data[0:trainingEnd]]
    validationData = [exampleType(res, slice) for res in data[trainingEnd:validationEnd]]
    testData = [exampleType(res, slice) for res in data[validationEnd:dataSize]]
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
    lossResult = sess.run(flagFieldNN.loss, getFeedDict(flagFieldNN, validationData))
    print("Validation loss: {}".format(lossResult))
