import karman
import TrainingConfiguration
import utils

trainConfig = TrainingConfiguration.TrainingConfiguration(NumObsPosX=30, NumObsPosY=20)
karman.generateTrainingExamples(trainConfig)
utils.serialize(trainConfig.simPath + "trainConfig.p", trainConfig)