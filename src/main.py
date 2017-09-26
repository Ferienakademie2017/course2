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
lossLogger = utils.LossLogger()
trainingData, validationData, testData = dataPartition.computeData(data,
                                                                   exampleType=evaluation.MultiStepSimulationCollection,
                                                                   slice=[0, 1], scale=1)

sess = tf.Session()


model1 = models.computeMultipleTimeStepNN1(1, reuse=False)
model2 = models.computeMultipleTimeStepNN1(2, reuse=True)
model4 = models.computeMultipleTimeStepNN1(4, reuse=True)
model8 = models.computeMultipleTimeStepNN1(8, reuse=True)
model16 = models.computeMultipleTimeStepNN1(16, reuse=True)

init = tf.global_variables_initializer()
sess.run(init)

model1.load(sess, "final")

def trainMultistep(model, multiStepSize, minibatchSize=10, numMinibatches=200):
    global trainingData
    global validationData
    global testData
    processedTrainingData = evaluation.generateMultiTimeStepExamples(trainingData, multiStepSize)
    #processedValidationData = evaluation.generateMultiTimeStepExamples(validationData, multiStepSize)
    #processedTestData = evaluation.generateMultiTimeStepExamples(testData, multiStepSize)

    training.trainNetwork(model, training.MinibatchSampler(processedTrainingData), lossLogger, minibatchSize, numMinibatches, sess=sess)

# Save final variables


#trainMultistep(sess, 1)
#trainMultistep(model1, 1, numMinibatches=200)
trainMultistep(model2, 2, numMinibatches=100)
trainMultistep(model4, 4, numMinibatches=50)
trainMultistep(model8, 8, numMinibatches=25)
trainMultistep(model16, 16, numMinibatches=15)

model1.save(sess, "final")

# Save the result of the loss logger
lossLogger.save()
#evaluation.validateModel(model, validationData, "final")
