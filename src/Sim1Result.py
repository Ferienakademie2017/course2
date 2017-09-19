import numpy


class Sim1Result():
    """Sim1Resluts is a class that holds the trainingdata for the first simple example
    npVel is a numpyarray(float) containing a velocity-field
    obstacle_pos is a list containing the position of an obstacle"""
    def __init__(self,npVel,obstacle_pos):
        self.npVel = npVel
        self.obstacle_pos = obstacle_pos


