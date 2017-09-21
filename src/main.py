import TrainingConfiguration
import training
import numpy as np
import scipy.ndimage
import utils
import evaluation
import models

trainConfig = utils.deserialize("data/trainConfig.p")
# data = []  # todo
data = trainConfig.loadGeneratedData()

trainingData, validationData, testData = evaluation.generateParametricExamples(data, 0.6, 0.4)
model = models.computeNN1()
minibatchSize = 8
training.trainNetwork(model, training.MinibatchSampler(trainingData), utils.LossLogger(), minibatchSize)
evaluation.validateModel(model, validationData)
