import karman
# import karmanInitialCond
import TrainingConfiguration
import utils

trainConfig = TrainingConfiguration.TrainingConfiguration(NumObsPosX=3, NumObsPosY=3, simPath='data/guiTest/', GUI=True)
karman.generateTrainingExamples(trainConfig)
#karmanInitialCond.generateTrainingExamples(trainConfig, numpy.concatenate((numpy.ones((32,64,1), dtype='f'),numpy.zeros((32,64,1), dtype='f')),axis = 2))

utils.serialize(trainConfig.simPath + "trainConfig.p", trainConfig)