import TrainingConfiguration
import Sim1Result
import training
import numpy as np
import tensorflow as tf
import scipy.ndimage
import utils
import evaluation
import models

def generateImgs(sess, model, folder, examples):
    folder = "images/" + folder
    # examples = [evaluation.TimeStepSimulationExample(outputManta, slice=[0, 1], scale=1) for outputManta in data]
    results = sess.run(model.yPred, feed_dict=evaluation.getFeedDict(model, examples))
    for e, r in zip(examples, results):
        outputTensor = Sim1Result.Sim1Result(r, [0], e.x[:,:,2], time=0)
        outputManta = Sim1Result.Sim1Result(e.y, [0], e.x[:,:,2], time=0)

        utils.sim1resToImage(outputManta, folder=folder)
        utils.sim1resToImage(outputTensor, folder= folder)
        utils.sim1resToImage(outputTensor, background='error', origRes=outputManta, folder=folder)


trainConfig = utils.deserialize("data/test_timeStep/trainConfig.p")
dataPartition = utils.deserialize(trainConfig.simPath + "dataPartition.p")
data = trainConfig.loadGeneratedData()
trainingData, validationData, testData = dataPartition.computeData(data, exampleType=evaluation.TimeStepSimulationCollection, slice=[0, 1], scale=1)
trainingData = evaluation.generateTimeStepExamples(trainingData)
validationData = evaluation.generateTimeStepExamples(validationData)
testData = evaluation.generateTimeStepExamples(testData)

model = models.computeTimeStepNN1()
init = tf.global_variables_initializer()
sess = tf.Session()
#sess.run(init)

# Load final variables
model.load(sess, "final")

numImages = 20
if(len(trainingData) >= numImages):
    generateImgs(sess, model, "training/", trainingData[:numImages])
if(len(validationData) >= numImages):
    generateImgs(sess, model, "validation/", validationData[:numImages])
if(len(testData) >= numImages):
    generateImgs(sess, model, "test/", testData[:numImages])
