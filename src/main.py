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

trainingData, validationData, testData = evaluation.generateParametricExamples(data)
model = models.computeNN1()
minibatchSize = 8
lossLogger = utils.LossLogger()
sess = training.trainNetwork(model, training.MinibatchSampler(trainingData), lossLogger, minibatchSize)

# Save final variables
model.save(sess, "final")
# Save the result of the loss logger
lossLogger.save()
# evaluation.validateModel(model, validationData)
