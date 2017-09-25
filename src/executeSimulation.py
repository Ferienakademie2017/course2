import karman
import TrainingConfiguration
import utils
import ObstacleContainer
import numpy as np

trainConfig = TrainingConfiguration.TrainingConfiguration(NumObsPosX=100, NumObsPosY=10, simPath='data/rand1/', GUI=False,maxObstacleNumber = 5,maxObstacleSize = 0.2,resY = 32,NumSteps=100)
initialCond = np.concatenate((np.concatenate((0.5*np.ones((32,1,1), dtype='f'),np.zeros((32,63,1), dtype='f')),axis = 1),np.zeros((32,64,1), dtype='f')),axis = 2)
#karman.generateTrainingExamples(trainConfig, initialCond, ObstacleContainer.simpleCylinder)
karman.generateTrainingExamples(trainConfig, initialCond, ObstacleContainer.generateObstacleContainer)


utils.serialize(trainConfig.simPath + "trainConfig.p", trainConfig)