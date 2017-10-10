import TrainingConfiguration
import training
import numpy as np
import scipy.ndimage
import utils
import evaluation
import models
import random
import tensorflow as tf

trainConfig = utils.deserialize("data/timeStep128x128/trainConfig.p")
# data = []  # todo
print("Load data")

data = trainConfig.loadGeneratedData()
dataPartition = evaluation.DataPartition(len(data), 0.8, 0.2)
utils.serialize(trainConfig.simPath + "dataPartition.p", dataPartition)

#trainingData, validationData, testData = dataPartition.computeData(data, exampleType=evaluation.FlagFieldSimulationExample, slice=[0, 1], scale=1)
lossLogger = utils.LossLogger()

print("Split data")

trainingData, validationData, testData = dataPartition.computeData(data,
                                                                   exampleType=evaluation.MultiStepSimulationCollection,
                                                                   slice=[0, 1], scale=1)

print("Data split")

def trainMultistep(multiStepSize, trainer, minibatchSize=10, numMinibatches=200):
    global trainingData

    print("process training data")
    processedTrainingData = evaluation.generateMultiTimeStepExamples(trainingData, multiStepSize)

    print("train")
    trainer.train(training.MinibatchSampler(processedTrainingData), lossLogger, minibatchSize, numMinibatches)

# Save final variables

def multiTrain():
    sess = tf.Session()

    multistepSizes = [1, 2, 4, 8, 16]
    minibatchCounts = [200, 100, 50, 50, 50]
    optimizerRates = [0.005, 0.002, 0.001, 0.0005, 0.0005]

    if len(multistepSizes) != len(minibatchCounts):
        raise ValueError("multiTrain(): len(multistepSizes) != len(minibatchCounts)")

    if len(multistepSizes) != len(optimizerRates):
        raise ValueError("multiTrain(): len(multistepSizes) != len(optimizerRates)")

    print("compute models")

    nnModels = [models.computeMultipleTimeStepNN3(n, reuse=(i != 0)) for i, n in enumerate(multistepSizes)]

    print("compute trainers")

    trainers = [training.NetworkTrainer(sess, model, learningRate) for model, learningRate in
                zip(nnModels, optimizerRates)]

    print("initialize variables")

    init = tf.global_variables_initializer()
    sess.run(init)

    print("load network")

    nnModels[0].load(sess, "multistep-padding")

    for i in range(len(multistepSizes)):
        trainMultistep(multistepSizes[i], trainers[i], numMinibatches=minibatchCounts[i])
    # file_writer = tf.summary.FileWriter('logs', sess.graph)
    nnModels[0].save(sess, "multistep-padding")

    global validationData
    processedValidationData = evaluation.generateMultiTimeStepExamples(validationData, multistepSizes[-1])
    evaluation.validateModel(nnModels[-1], processedValidationData, "multistep-padding")


def autoencoderTrain():
    global trainingData
    global lossLogger
    processedTrainingData = evaluation.generateAutoencoderExamples(trainingData)
    autoencoder = models.computeAutoencoderNN1()
    autodecoder = models.computeAutodecoderNN1()
    sess = tf.Session()
    trainer = training.NetworkTrainer(sess, autoencoder)
    init = tf.global_variables_initializer()
    sess.run(init)

    trainer.train(training.MinibatchSampler(processedTrainingData), lossLogger, minibatchSize=20, numMinibatches=200)
    # file_writer = tf.summary.FileWriter('logs', sess.graph)
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
    
    # file_writer = tf.summary.FileWriter('logs', sess.graph)

    model.save(sess, "timeStep")

def trainAutoStep(multiStepSize, trainer, sess, autoencoderModel, minibatchSize=10, numMinibatches=200):
    global trainingData

    processedTrainingData = evaluation.generateAutoStepExamples(trainingData, multiStepSize, sess, autoencoderModel)

    trainer.train(training.MinibatchSampler(processedTrainingData), lossLogger, minibatchSize, numMinibatches)

def autoStepTrain():
    sess = tf.Session()

    multistepSizes = [1, 2, 4, 8, 16]
    minibatchCounts = [200, 200, 200, 200, 200]
    optimizerRates = [0.005, 0.003, 0.0015, 0.0008, 0.0005]

    if len(multistepSizes) != len(minibatchCounts):
        raise ValueError("multiTrain(): len(multistepSizes) != len(minibatchCounts)")

    if len(multistepSizes) != len(optimizerRates):
        raise ValueError("multiTrain(): len(multistepSizes) != len(optimizerRates)")

    autoencoderModel = models.computeAutoencoderNN1()
    autoencoderModel.load(sess, "autoencoder")

    nnModels = [models.computeAutoStepNN1(n, reuse=(i != 0)) for i, n in enumerate(multistepSizes)]

    trainers = [training.NetworkTrainer(sess, model, learningRate) for model, learningRate in zip(nnModels, optimizerRates)]

    init = tf.global_variables_initializer()
    sess.run(init)

    # nnModels[0].load(sess, "autoStep")

    for i in range(len(multistepSizes)):
        trainAutoStep(multistepSizes[i], trainers[i], sess, autoencoderModel, numMinibatches=minibatchCounts[i])
    # file_writer = tf.summary.FileWriter('logs', sess.graph)
    nnModels[0].save(sess, "autoStep")


# multiTrain()
# autoencoderTrain()
# timeStepTrain()
autoStepTrain()

# Save the result of the loss logger
lossLogger.save()
#evaluation.validateModel(model, validationData, "final")
