import TrainingConfiguration
import training
import numpy as np
import scipy.ndimage
import utils
import evaluation
import models
import random

trainConfig = utils.deserialize("data/rand1/trainConfig.p")
# data = []  # todo
data = trainConfig.loadGeneratedData()
dataPartition = evaluation.DataPartition(len(data), 0.6, 0.4)
trainingData, validationData, testData = dataPartition.computeData(data, exampleType=evaluation.FlagFieldSimulationExample, slice=[0, 1], scale=1)
utils.serialize(trainConfig.simPath + "dataPartition.p", dataPartition)

model = models.computeNN9()
minibatchSize = 10
lossLogger = utils.LossLogger()
sess = training.trainNetwork(model, training.MinibatchSampler(trainingData), lossLogger, minibatchSize, 400)

# Save final variables
model.save(sess, "final")
# Save the result of the loss logger
lossLogger.save()
evaluation.validateModel(model, validationData, "final")
