import karman
import TrainingConfiguration
import utils

trainConfig = TrainingConfiguration.TrainingConfiguration()
karman.generateTrainingExamples(trainConfig)
utils.serialize(trainConfig.savedata + "trainConfig.p", trainConfig)