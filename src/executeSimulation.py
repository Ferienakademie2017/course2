import karman
import TrainingConfiguration
import utils

trainConfig = TrainingConfiguration.TrainingConfiguration()
karman.generateTrainingExamples(trainConfig)
utils.serialize(trainConfig.simPath + "trainConfig.p", trainConfig)