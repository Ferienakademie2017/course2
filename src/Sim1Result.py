import numpy as np

class Sim1Result(object):
    """Sim1Resluts is a class that holds the trainingdata for the first simple example
    npVel is a numpyarray(float) containing a velocity-field
    obstacle_pos is a list containing the position of an obstacle"""
    def __init__(self, npVel, obstacle_pos, obstacles):
        self.npVel = npVel
        self.obstacle_pos = obstacle_pos
        self.obstacles = obstacles
