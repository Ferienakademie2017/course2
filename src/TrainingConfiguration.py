import utils

class TrainingConfiguration(object):
    def __init__(self,simPath = 'data/',savedata = True,saveppm = False,NumObsPosX = 1,NumObsPosY = 10,GUI = False,resY = 32,resX = 64,saveInterval = 100,NumSteps=100,maxObstacleNumber = 1,maxObstacleSize = 0.2,timeOffset = 100):
        self.simPath = simPath
        self.savedata = savedata
        self.saveppm = saveppm
        self.NumObsPosX = NumObsPosX
        self.NumObsPosY = NumObsPosY
        self.GUI = GUI
        self.resY = resY
        self.resX = resX
        self.saveInterval = saveInterval
        self.NumSteps = NumSteps
        self.maxObstacleNumber = maxObstacleNumber
        self.maxObstacleSize = maxObstacleSize
        self.timeOffset = timeOffset

    def getFileNameFor(self,simNo,stepNo):
        return 'vel_SimNo{}_stepNo{}.p'.format(simNo,stepNo)

    def loadGeneratedData(self):
        result_List = []
        for simNo in range(0,self.NumObsPosX*self.NumObsPosY):
            for saveNo in range(0,(self.NumSteps//self.saveInterval)+1):
                result_List.append(utils.deserialize(self.simPath + self.getFileNameFor(simNo,saveNo*self.saveInterval)))
        return result_List


