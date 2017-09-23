import karman
import TrainingConfiguration
import utils
import ObstacleContainer
import numpy as np

trainConfig = TrainingConfiguration.TrainingConfiguration(NumObsPosX=10, NumObsPosY=10, simPath='data/guiTest/', GUI=True,maxObstacleNumber = 5,maxObstacleSize = 0.2,resY = 64,NumSteps=100)
initialCond = np.concatenate((np.concatenate((0.5*np.ones((1,64,1), dtype='f'),np.zeros((63,64,1), dtype='f')),axis = 0),np.zeros((64,64,1), dtype='f')),axis = 2)
karman.generateTrainingExamples(trainConfig, initialCond,ObstacleContainer.simpleCylinder)
#karman.generateTrainingExamples(trainConfig, initialCond,ObstacleContainer.generateObstacleContainer)


utils.serialize(trainConfig.simPath + "trainConfig.p", trainConfig)