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

class TimeStepSimulationExample(object):
    def __init__(self, x, y, flagField):
        self.x = np.concatenate((x, np.expand_dims(flagField, -1)), -1)
        self.y = y
        self.flagField = flagField

class AutoStepSimulationExample(object):
    def __init__(self, x, y, flagField, sess, autoencoder):
        self.x = sess.run(autoencoder.encoding, {autoencoder.x: x, autoencoder.flagField: flagField})
        self.y = sess.run(autoencoder.encoding, {autoencoder.x: y, autoencoder.flagField: flagField})
        self.flagField = flagField

class AutoencoderExample(object):
    def __init__(self, x, flagField):
        self.x = x #np.concatenate((x, np.expand_dims(flagField, -1)), -1)
        self.y = self.x
        self.flagField = flagField

class TimeStepSimulationCollection(object):
    def __init__(self, sim1ResultList, slice=[1], scale=0.25):
        # print(sim1Result.npVel.shape)
        self.velFields = []
        for res in sim1ResultList:
            arr = res.npVel
            arr = np.delete(arr, 2, 2)
            self.velFields.append(arr)

        func = lambda x: 1.0 if x > 0.0 else 0.0
        obs = sim1ResultList[0].obstacles
        obs = np.vectorize(func)(obs)
        self.flagField = obs
        self.obstacles = sim1ResultList[0].obstacles,

    def getExamples(self):
        examples = []
        for i in range(len(self.velFields) - 1):
            examples.append(TimeStepSimulationExample(self.velFields[i], self.velFields[i+1], self.flagField))
        return examples


def generateTimeStepExamples(collectionList):
    """Input: a list of TimeStepSimulationCollection objects, generated by a DataPartition"""
    result = []
    for l in collectionList:
        result.extend(l.getExamples())
    return result

class MultiStepSimulationCollection(object):  # todo: join with TimeStepSimulationCollection
    def __init__(self, sim1ResultList, slice=[1], scale=0.25,numTimeSteps = 1):
        # print(sim1Result.npVel.shape)
        self.velFields = []
        self.numTimeSteps = numTimeSteps
        for res in sim1ResultList:
            arr = res.npVel
            arr = np.delete(arr, 2, 2)
            self.velFields.append(arr)

        func = lambda x: 1.0 if x > 0.0 else 0.0
        obs = sim1ResultList[0].obstacles
        obs = np.vectorize(func)(obs)
        self.flagField = obs
        self.obstacles = sim1ResultList[0].obstacles

    def getExamples(self,numTimeSteps):
        examples = []
        self.numTimeSteps = numTimeSteps
        for i in range(0, len(self.velFields) - self.numTimeSteps, max(1, self.numTimeSteps/2)):
            y = np.concatenate([np.expand_dims(self.velFields[i+n],-1) for n in range(self.numTimeSteps)],-1)
            examples.append(TimeStepSimulationExample(self.velFields[i],y, self.flagField))
        return examples

    def getAutoencoderExamples(self):
        return [AutoencoderExample(field, self.flagField) for field in self.velFields]

    def getAutoStepExamples(self, numTimeSteps, sess, model):
        examples = []
        self.numTimeSteps = numTimeSteps
        for i in range(0, len(self.velFields) - self.numTimeSteps, max(1, self.numTimeSteps / 2)):
            y = np.concatenate([np.expand_dims(self.velFields[i + n], -1) for n in range(self.numTimeSteps)], -1)
            examples.append(AutoStepSimulationExample(self.velFields[i], y, self.flagField, sess, model))
        return examples


def generateAutoencoderExamples(collectionList):
    result = []
    for l in collectionList:
        result.extend(l.getAutoencoderExamples())
    return result

def generateMultiTimeStepExamples(collectionList,numTimeSteps):
    """Input: a list of MultiTimeStepSimulationCollection objects, generated by a DataPartition"""
    result = []
    for l in collectionList:
        result.extend(l.getExamples(numTimeSteps))
    return result

class DataPartition(object):
    def __init__(self, dataSize, trainingFraction=0.6, validationFraction=0.2):
        self.dataSize = dataSize
        self.trainingFraction = trainingFraction
        self.validationFraction = validationFraction
        indices = [i for i in range(dataSize)]
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

def getFeedDict(network, data, isTraining):
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
    return {network.x: xValues, network.y: yValues, network.flagField: ffValues, network.phase: isTraining}

def validateModel(flagFieldNN, validationData, name="final"):
    sess = tf.Session()
    flagFieldNN.load(sess, name)
    yPred, lossResult = sess.run([flagFieldNN.yPred, flagFieldNN.loss], getFeedDict(flagFieldNN, validationData, isTraining=False))
    print("Validation loss: {}".format(lossResult))
