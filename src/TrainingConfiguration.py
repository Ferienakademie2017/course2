

class TrainingConfiguration():


    def __init__(self,simPath = 'data/',savedata = True,saveppm = False,NumObsPosX = 1,NumObsPosY = 10,GUI = False,resY = 32,resX = 64,stepIntervall = 100):
        self.simPath = simPath
        self.savedata = savedata
        self.saveppm = saveppm
        self.NumObsPosX = NumObsPosX
        self.NumObsPosY = NumObsPosY
        self.GUI = GUI
        self.resY = resY
        self.resX = resX
        self.stepIntervall = stepIntervall


