import TrainingConfiguration
import Sim1Result
import training
import numpy as np
import tensorflow as tf
import scipy.ndimage
import utils
import evaluation
import models

def generateImgs(sess, model, folder, data):
    examples = [evaluation.FlagFieldSimulationExample(outputManta, slice=[0, 1], scale=1) for outputManta in data]
    results = sess.run(model.yPred, feed_dict=evaluation.getFeedDict(model, examples))
    for d, r in zip(data, results):
        outputTensor = Sim1Result.Sim1Result(r, d.obstacle_pos, d.obstacles)

        utils.sim1resToImage(d)
        utils.sim1resToImage(outputTensor)
        utils.sim1resToImage(outputTensor, background='error', origRes=d)


trainConfig = utils.deserialize("data/rand1/trainConfig.p")
dataPartition = utils.deserialize(trainConfig.simPath + "dataPartition.p")
data = trainConfig.loadGeneratedData()
trainingData, validationData, testData = dataPartition.computeData(data, exampleType=evaluation.FlagFieldSimulationExample, slice=[0, 1], scale=1)

model = models.computeNN9()
init = tf.global_variables_initializer()
sess = tf.Session()
#sess.run(init)

# Load final variables
model.load(sess, "final")

generateImgs(sess, model, "training", trainingData[:5])
generateImgs(sess, model, "validation", validationData[:5])
generateImgs(sess, model, "test", testData[:5])
