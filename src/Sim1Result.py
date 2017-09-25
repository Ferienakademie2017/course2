import numpy


class Sim1Result(object):
    """Sim1Resluts is a class that holds the trainingdata for the first simple example
    npVel is a numpyarray(float) containing a velocity-field
    obstacle_pos is a list containing the position of an obstacle"""
    def __init__(self):
        self.timeStep = []
        self.npVel = []
        self.obstacle_pos = []
        self.obstacles = []

    def addValue(self,npVel, obstacle_pos, obstacles,timeStep):
        self.timeStep.append(timeStep)
        self.npVel.append(npVel)
        self.obstacle_pos.append(obstacle_pos)
        self.obstacles.append(obstacles)
