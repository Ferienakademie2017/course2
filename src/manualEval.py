import TrainingConfiguration
import Sim1Result
import training
import numpy as np
import tensorflow as tf
import scipy.ndimage
import utils
import evaluation
import models
import random

def generateMultiSequence(sess, model, folder, example, numSteps=50):
    folder = "images/" + folder

    initialCond = example.x
    example.y = np.expand_dims(example.y, -1)

    for i in range(numSteps):
        result = Sim1Result.Sim1Result(example.x[:,:,0:2], [0], example.x[:,:,2], time=0)
        utils.sim1resToImage(result, folder=folder)
        newResult = sess.run(model.yPred, evaluation.getFeedDict(model, [example], isTraining=False))
        example.x[:,:,0:2] = newResult[0][:,:,:,0]
        example.x[:,:,0:2] = example.x[:,:,0:2] * example.x[:,:,2:3]
        example.x[0,:,:] = initialCond[0,:,:]

def generateSequence(sess, model, folder, example, numSteps=50):
    folder = "images/" + folder

    initialCond = example.x

    for i in range(numSteps):
        result = Sim1Result.Sim1Result(example.x[:,:,0:2], [0], example.x[:,:,2], time=0)
        utils.sim1resToImage(result, folder=folder)
        newResult = sess.run(model.yPred, evaluation.getFeedDict(model, [example], isTraining=False))
        example.x[:,:,0:2] = newResult[0]
        example.x[:,:,0:2] = example.x[:,:,0:2] * example.x[:,:,2:3]
        example.x[0,:,:] = initialCond[0,:,:]
        #for i in range(1, 64):
        #    example.x[i,:,0] *= (initialFlow + 0.01) / (sum(example.x[i,:,0]) + 0.01)


def generateImgs(sess, model, folder, examples):
    folder = "images/" + folder
    # examples = [evaluation.TimeStepSimulationExample(outputManta, slice=[0, 1], scale=1) for outputManta in data]
    results = sess.run(model.yPred, feed_dict=evaluation.getFeedDict(model, examples, isTraining=False))
    for e, r in zip(examples, results):
        outputTensor = Sim1Result.Sim1Result(r, [0], e.x[:,:,2], time=0)
        outputManta = Sim1Result.Sim1Result(e.y, [0], e.x[:,:,2], time=0)

        utils.sim1resToImage(outputManta, folder=folder)
        utils.sim1resToImage(outputTensor, folder=folder)
        utils.sim1resToImage(outputTensor, background='error', origRes=outputManta, folder=folder)


trainConfig = utils.deserialize("data/test_timeStep/trainConfig.p")
dataPartition = utils.deserialize(trainConfig.simPath + "dataPartition.p")
data = trainConfig.loadGeneratedData()
trainingData, validationData, testData = dataPartition.computeData(data, exampleType=evaluation.TimeStepSimulationCollection, slice=[0, 1], scale=1)
trainingData = evaluation.generateTimeStepExamples(trainingData)
validationData = evaluation.generateTimeStepExamples(validationData)
testData = evaluation.generateTimeStepExamples(testData)

model = models.computeMultipleTimeStepNN1(1)
init = tf.global_variables_initializer()
sess = tf.Session()
#sess.run(init)

# Load final variables
model.load(sess, "final")

def randomSample(list, numSamples):
    """with replacement"""
    return [list[i] for i in sorted(random.sample(xrange(len(list)), numSamples))]


# numImages = 20
# if len(trainingData) >= 1:
#     generateImgs(sess, model, "training/", randomSample(trainingData, numImages))
# if len(validationData) >= 1:
#     generateImgs(sess, model, "validation/", randomSample(validationData, numImages))
# if len(testData) >= 1:
#     generateImgs(sess, model, "test/", randomSample(testData, numImages))

generateMultiSequence(sess, model, "sequenceMulti01/", validationData[0])
generateMultiSequence(sess, model, "sequenceMulti02/", validationData[100])
