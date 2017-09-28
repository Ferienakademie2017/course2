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

def trainMultistep(multiStepSize, trainer, minibatchSize=10, numMinibatches=200):
    global trainingData
    global validationData
    global testData
    processedTrainingData = evaluation.generateMultiTimeStepExamples(trainingData, multiStepSize)
    #processedValidationData = evaluation.generateMultiTimeStepExamples(validationData, multiStepSize)
    #processedTestData = evaluation.generateMultiTimeStepExamples(testData, multiStepSize)

    trainer.train(training.MinibatchSampler(processedTrainingData), lossLogger, minibatchSize, numMinibatches)

# Save final variables

def multiTrain():
    sess = tf.Session()

    multistepSizes = [1, 2, 4, 8, 16]
    minibatchCounts = [200, 100, 50, 50, 50]

    if len(multistepSizes) != len(minibatchCounts):
        raise ValueError("multiTrain(): len(multistepSizes) != len(minibatchCounts)")

    nnModels = [models.computeMultipleTimeStepNN1(n, reuse=(i != 0)) for i, n in enumerate(multistepSizes)]

    trainers = [training.NetworkTrainer(sess, model) for model in nnModels]

    init = tf.global_variables_initializer()
    sess.run(init)

    # model1.load(sess, "multistep")

    for i in range(len(multistepSizes)):
        trainMultistep(multistepSizes[i], trainers[i], numMinibatches=minibatchCounts[i])

    models[0].save(sess, "multistep")


def autoencoderTrain():
    global trainingData
    global lossLogger
    processedTrainingData = evaluation.generateAutoencoderExamples(trainingData)
    autoencoder = models.computeAutoencoderNN1()
    sess = tf.Session()
    trainer = training.NetworkTrainer(sess, autoencoder)
    init = tf.global_variables_initializer()
    sess.run(init)

    trainer.train(training.MinibatchSampler(processedTrainingData), lossLogger, minibatchSize=20, numMinibatches=200)

    autoencoder.save(sess, "autoencoder")


def timeStepTrain():
    global trainingData
    processedTrainingData = evaluation.generateTimeStepExamples(trainingData)
    sess = tf.Session()

    model = models.computeTimeStepNN1()
    trainer = training.NetworkTrainer(sess, model)

    init = tf.global_variables_initializer()
    sess.run(init)

    trainer.train(training.MinibatchSampler(processedTrainingData), lossLogger, minibatchSize=20, numMinibatches=200)

    model.save(sess, "timeStep")

multiTrain()
# autoencoderTrain()
# timeStepTrain()

# Save the result of the loss logger
lossLogger.save()
#evaluation.validateModel(model, validationData, "final")
