import TrainingConfiguration
import training
import numpy as np
import scipy.ndimage
import utils
import evaluation
import models

trainConfig = utils.deserialize("data/trainConfig.p")
data = []  # todo
# data = trainConfig.loadData()

trainingData, validationData, testData = evaluation.generateParametricExamples(data)
training.trainNetwork(models.computeNN1(), training.MinibatchSampler(trainingData))





