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

    model1 = models.computeMultipleTimeStepNN1(1, reuse=False)
    model2 = models.computeMultipleTimeStepNN1(2, reuse=True)
    model4 = models.computeMultipleTimeStepNN1(4, reuse=True)
    model8 = models.computeMultipleTimeStepNN1(8, reuse=True)
    model16 = models.computeMultipleTimeStepNN1(16, reuse=True)

    trainer1 = training.NetworkTrainer(sess, model1)
    trainer2 = training.NetworkTrainer(sess, model2)
    trainer4 = training.NetworkTrainer(sess, model4)
    trainer8 = training.NetworkTrainer(sess, model8)
    trainer16 = training.NetworkTrainer(sess, model16)

    init = tf.global_variables_initializer()
    sess.run(init)

    # model1.load(sess, "multistep")

    trainMultistep(1, trainer1, numMinibatches=200)
    trainMultistep(2, trainer2, numMinibatches=100)
    trainMultistep(4, trainer4, numMinibatches=50)
    trainMultistep(8, trainer8, numMinibatches=25)
    trainMultistep(16, trainer16, numMinibatches=15)

    model1.save(sess, "multistep")


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
