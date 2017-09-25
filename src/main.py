import TrainingConfiguration
import training
import numpy as np
import scipy.ndimage
import utils
import evaluation
import models
import random

trainConfig = utils.deserialize("data/test_timeStep/trainConfig.p")
# data = []  # todo
data = trainConfig.loadGeneratedData()
dataPartition = evaluation.DataPartition(len(data), 0.6, 0.4)
utils.serialize(trainConfig.simPath + "dataPartition.p", dataPartition)

#trainingData, validationData, testData = dataPartition.computeData(data, exampleType=evaluation.FlagFieldSimulationExample, slice=[0, 1], scale=1)

trainingData, validationData, testData = dataPartition.computeData(data, exampleType=evaluation.TimeStepSimulationCollection, slice=[0, 1], scale=1)
trainingData = evaluation.generateTimeStepExamples(trainingData)
validationData = evaluation.generateTimeStepExamples(validationData)
testData = evaluation.generateTimeStepExamples(testData)

model = models.computeTimeStepNN1()
minibatchSize = 10
numMinibatches = 200
lossLogger = utils.LossLogger()
sess = training.trainNetwork(model, training.MinibatchSampler(trainingData), lossLogger, minibatchSize, numMinibatches)

# Save final variables
model.save(sess, "final")
# Save the result of the loss logger
lossLogger.save()
evaluation.validateModel(model, validationData, "final")
