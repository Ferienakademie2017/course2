import TrainingConfiguration
import Sim1Result
import training
import numpy as np
import tensorflow as tf
import scipy.ndimage
import utils
import evaluation
import models

trainConfig = utils.deserialize("data/trainConfig.p")
# data = []  # todo
data = trainConfig.loadGeneratedData()

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
model = models.computeNN5()

# Load final variables
model.load(sess, "final")

# Sample some input data
outputManta = data[len(data) // 2 + 1]
outputManta.npVel = np.transpose(outputManta.npVel, (1, 0, 2))
outputManta.obstacles = np.transpose(outputManta.obstacles)
manualResults = sess.run(model.yPred, evaluation.getFeedDict(model, [evaluation.ParametricSimulationExample(outputManta, slice=[0, 1])]))

outputTensor = Sim1Result.Sim1Result(manualResults[0], outputManta.obstacle_pos, outputManta.obstacles)
# outputTensor.npVel = np.transpose(outputTensor.npVel, (1, 0, 2))
# outputTensor.obstacles = np.transpose(outputTensor.obstacles)
# outputManta.npVel = np.transpose(outputManta.npVel, (1, 0, 2))
# outputManta.obstacles = np.transpose(outputManta.obstacles)

utils.image_i = 100
utils.sim1resToImage(outputManta)
utils.sim1resToImage(outputTensor)
