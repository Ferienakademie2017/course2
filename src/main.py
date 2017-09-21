import TrainingConfiguration
import training
import numpy as np
import scipy.ndimage
import utils
import evaluation
import models
import random

trainConfig = utils.deserialize("data/trainConfig.p")
# data = []  # todo
data = trainConfig.loadGeneratedData()
random.shuffle(data)

trainingData, validationData, testData = evaluation.generateParametricExamples(data, 0.6, 0.4, slice=[0, 1])
model = models.computeNN5()
minibatchSize = 100
lossLogger = utils.LossLogger()
sess = training.trainNetwork(model, training.MinibatchSampler(trainingData), lossLogger, minibatchSize)

# Save final variables
model.save(sess, "final")
# Save the result of the loss logger
lossLogger.save()
evaluation.validateModel(model, validationData, "final")
