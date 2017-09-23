import TrainingConfiguration
import Sim1Result
import training
import numpy as np
import tensorflow as tf
import scipy.ndimage
import utils
import evaluation
import models

trainConfig = utils.deserialize("data/rand1/trainConfig.p")
# data = []  # todo
data = trainConfig.loadGeneratedData()[0:200]

model = models.computeNN9()
init = tf.global_variables_initializer()
sess = tf.Session()
#sess.run(init)

# Load final variables
model.load(sess, "final")

# Sample some input data
# outputManta = data[18]
utils.image_i = 10000
#examples = [evaluation.ParametricSimulationExample(outputManta, slice=[0, 1], scale=1) for outputManta in data]
examples = [evaluation.FlagFieldSimulationExample(outputManta, slice=[0, 1], scale=1) for outputManta in data]
results = sess.run(model.yPred, feed_dict=evaluation.getFeedDict(model, examples))


for i in range(len(data)):
    #outputManta = data[i]
    #simEx = evaluation.ParametricSimulationExample(outputManta, slice=[0, 1], scale=0.25)
    # outputManta.npVel = np.transpose(outputManta.npVel, (1, 0, 2))
    # outputManta.obstacles = np.transpose(outputManta.obstacles)
    #manualResults = sess.run(model.yPred, feed_dict=evaluation.getFeedDict(model, [simEx]))
    #print(sum(sum(sum(manualResults[0]))))

    outputTensor = Sim1Result.Sim1Result(results[i], data[i].obstacle_pos, data[i].obstacles)

    utils.sim1resToImage(data[i])
    utils.sim1resToImage(outputTensor)
