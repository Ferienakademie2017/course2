import karman
import TrainingConfiguration
import utils
import ObstacleContainer
import numpy as np

resY = 128
resX = 128
#trainConfig = TrainingConfiguration.TrainingConfiguration(NumObsPosX=40, NumObsPosY=1, simPath='data/timeStep128x128/',
#                                                          GUI=False, maxObstacleNumber = 5, maxObstacleSize = 0.2,
#                                                          resX = resX, resY = resY,  NumSteps=200,timeOffset = 0,
#                                                          saveInterval=4)
trainConfig = utils.deserialize("data/timeStep128x128/trainConfig.p")
initialCond = np.concatenate((np.concatenate((0.5*np.ones((resY, 1, 1), dtype='f'),np.zeros((resY, resX-1, 1), dtype='f')),axis = 1),np.zeros((resY,resX,1), dtype='f')),axis = 2)
#karman.generateTrainingExamples(trainConfig, initialCond, ObstacleContainer.simpleCylinder)
karman.generateTrainingExamples(trainConfig, initialCond, ObstacleContainer.generateObstacleContainer)


utils.serialize(trainConfig.simPath + "trainConfig.p", trainConfig)