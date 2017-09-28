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

def getInitialSmoke(shape, flagField):
    xSize = shape[0]
    ySize = shape[1]
    smokeField = np.zeros([xSize, ySize])
    for x in range(xSize):
        for y in range(ySize):
            if (float(x) / xSize - 0.1) ** 2 < 0.1 ** 2 \
                    and (float(y) / ySize - 0.5) ** 2 < 0.3 ** 2 \
                    and flagField[x, y] > 0.2:
                smokeField[x, y] = 1.0

    return smokeField

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

def generateMultiSequence2(sess, model, folder, example, numSteps=50):
    folder = "images/" + folder

    initialCond = example.x
    example.y = np.expand_dims(example.y, -1)

    smokeField = getInitialSmoke(example.x.shape[0:2], example.flagField)

    for i in range(numSteps):
        result = Sim1Result.Sim1Result(example.x[:,:,0:2], [0], example.x[:,:,2], time=0)
        utils.sim1resToImage(result, folder=folder, smokeField = smokeField)
        for t in range(4):
            advect(smokeField, example.x[:, :, 0:2])
        newResult = sess.run(model.yPred, evaluation.getFeedDict(model, [example], isTraining=False))
        example.x[:,:,0:2] = newResult[0][:,:,:,0]
        example.x[:,:,0:2] = example.x[:,:,0:2] * example.x[:,:,2:3]
        example.x[0,:,:] = initialCond[0,:,:]


def advect(x,v):
    """v Geschwindigkeitsfeld"""
    dt = 1
    dims = v.shape
    gridPoints = np.meshgrid(range(dims[1]), range(dims[0]))
    gridPoints = np.concatenate((np.expand_dims(gridPoints[0], axis=-1), np.expand_dims(gridPoints[1], axis=-1)),
                                axis=-1)
    prevPointsDouble = gridPoints[:, :, [1, 0]] - dt * v[:, :, 0:2]
    return interpolate(prevPointsDouble, x)


def interpolate(prevPointsDouble, x):
    dims = x.shape
    for ind1 in range(dims[0]):
        for ind2 in range(dims[1]):
            if prevPointsDouble[ind1,ind2,0] < 0:
                prevPointsDouble[ind1, ind2, 0] = 0
            if prevPointsDouble[ind1,ind2,0] > dims[0]-2:
                prevPointsDouble[ind1, ind2, 0] = dims[0] -1.01
            if prevPointsDouble[ind1,ind2,1] > dims[1]-2:
                prevPointsDouble[ind1, ind2, 1] = dims[1] -1.01
            if prevPointsDouble[ind1, ind2, 1] < 0:
                prevPointsDouble[ind1, ind2, 1] = 0
    prevInt = prevPointsDouble.astype("int")
    x_new = np.zeros(dims,"float")
    for i1 in range(dims[0]):
        for i2 in range(dims[1]):
            a = prevInt[i1,i2,0]
            b = prevInt[i1,i2,1]
            if a >=31:
                a = a
            x1 = prevPointsDouble[i1,i2,0]-a
            x2 = prevPointsDouble[i1, i2, 1]-b
            x_new[i1,i2] = x[a, b] + (x[a, b + 1] - x[a, b]) * x2 + (x[a + 1, b] - x[a, b]) * x1 + (x[a + 1, b + 1] - x[a + 1, b] - x[a, b + 1] + x[a, b]) * x1 * x2

    return x_new

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


trainConfig = utils.deserialize("data/timeStep128x128/trainConfig.p")
dataPartition = utils.deserialize(trainConfig.simPath + "dataPartition.p")
data = trainConfig.loadGeneratedData()
trainingData, validationData, testData = dataPartition.computeData(data, exampleType=evaluation.TimeStepSimulationCollection, slice=[0, 1], scale=1)
trainingData = evaluation.generateTimeStepExamples(trainingData)
validationData = evaluation.generateTimeStepExamples(validationData)
testData = evaluation.generateTimeStepExamples(testData)

model = models.computeMultipleTimeStepNN3(1)
init = tf.global_variables_initializer()
sess = tf.Session()
#sess.run(init)

# Load final variables
model.load(sess, "multistep")

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

generateMultiSequence2(sess, model, "sequences/128_smoke_0/", validationData[0], numSteps=200)
generateMultiSequence2(sess, model, "sequences/128_smoke_1/", validationData[100], numSteps=200)
generateMultiSequence2(sess, model, "sequences/128_smoke_2/", validationData[200], numSteps=200)
