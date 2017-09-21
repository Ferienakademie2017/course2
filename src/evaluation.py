import tensorflow as tf
import numpy as np
import scipy.ndimage

class SimulationExample(object):
    def __init__(self, sim1Result):
        self.x = sim1Result.obstacle_pos
        np.delete(sim1Result.npVel, 2, 2)
        self.y = scipy.ndimage.zoom(sim1Result.npVel, [0.25, 0.25, 1])
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
