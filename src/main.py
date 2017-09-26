import TrainingConfiguration
import training
import numpy as np
import scipy.ndimage
import utils
import evaluation
import models
import random
import tensorflow as tf

trainConfig = utils.deserialize("data/test_timeStep/trainConfig.p")
# data = []  # todo
data = trainConfig.loadGeneratedData()
dataPartition = evaluation.DataPartition(len(data), 0.6, 0.4)
utils.serialize(trainConfig.simPath + "dataPartition.p", dataPartition)

#trainingData, validationData, testData = dataPartition.computeData(data, exampleType=evaluation.FlagFieldSimulationExample, slice=[0, 1], scale=1)
trainingData, validationData, testData = dataPartition.computeData(data, exampleType=evaluation.MultiStepSimulationCollection, slice=[0, 1], scale=1)
trainingData = evaluation.generateMultiTimeStepExamples(trainingData,5)
validationData = evaluation.generateMultiTimeStepExamples(validationData,5)
testData = evaluation.generateTimeStepExamples(testData)

model = models.computeMultipleTimeStepNN2(5)
minibatchSize = 10
numMinibatches = 200
lossLogger = utils.LossLogger()
sess = tf.Session()
sess = training.trainNetwork(sess, model, training.MinibatchSampler(trainingData), lossLogger, minibatchSize, numMinibatches)

# TensorBoard
file_writer = tf.summary.FileWriter('logs', sess.graph)

# Save final variables
model.save(sess, "final")
# Save the result of the loss logger
lossLogger.save()
evaluation.validateModel(model, validationData, "final")
