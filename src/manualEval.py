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
outputManta = data[len(data) // 2 + 2]
simEx = evaluation.ParametricSimulationExample(outputManta, slice=[0, 1])
print(simEx.flagField)
print("*****************")
print(outputManta.obstacles)
# outputManta.npVel = np.transpose(outputManta.npVel, (1, 0, 2))
# outputManta.obstacles = np.transpose(outputManta.obstacles)
manualResults, loss = sess.run([model.yPred, model.loss], evaluation.getFeedDict(model, [simEx]))

# Compute loss
outManta = scipy.ndimage.zoom(outputManta.npVel, [0.25, 0.25, 1])
outTensor = manualResults[0]
print("Manta: {}".format(outManta.shape))
print("Tensor: {}".format(outTensor.shape))
tempLoss = 0
for i in range(len(outManta)):
    for j in range(len(outManta[0])):
        for k in range(2):
            if outputManta.obstacles[i, j] > 0:
                tempLoss += abs(outManta[i, j, k] - outTensor[i, j, k])


print("Temploss: {}".format(tempLoss))
print("Loss: {}".format(loss))
outputTensor = Sim1Result.Sim1Result(manualResults[0], outputManta.obstacle_pos, outputManta.obstacles)
# outputTensor.npVel = np.transpose(outputTensor.npVel, (1, 0, 2))
# outputTensor.obstacles = np.transpose(outputTensor.obstacles)
# outputManta.npVel = np.transpose(outputManta.npVel, (1, 0, 2))
# outputManta.obstacles = np.transpose(outputManta.obstacles)

utils.image_i = 100
utils.sim1resToImage(outputManta)
utils.sim1resToImage(outputTensor)
