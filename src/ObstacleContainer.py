from manta import *
import random

def generateObstacleContainer(s,trainingConfiguration,simNo):
    ObstacleList = []
    for ind in range(0,trainingConfiguration.maxObstacleNumber):
        ObstacleList.append(generateObstacle(s,trainingConfiguration))
    return ObstacleList

def generateObstacle(s,trainingConfiguration):
    maxSize = trainingConfiguration.maxObstacleSize
    gs = vec3(trainingConfiguration.resX, trainingConfiguration.resY, trainingConfiguration.resY)
    res = trainingConfiguration.resY
    indexList = [0,1]
    ind = random.choice(indexList)
    pos = [random.random(), random.random(), 0.5]

    if ind == 0:
        sizeY = random.random()
        return Cylinder(parent=s, center=gs * vec3(sizeY*maxSize + (1-2*maxSize*sizeY) * pos[0], pos[1], pos[2]), radius=res * maxSize*sizeY,
                 z=gs * vec3(0, 0, 1.0))
    if ind == 1:
        sizeY = maxSize*random.random()
        sizeX = maxSize*random.random()
        return Box(parent=s, p0 = gs * vec3((1-2*sizeX)*pos[0], pos[1], 0), p1=gs *vec3((1-2*sizeX)*pos[0]+sizeX,pos[1]+sizeY,1))

def simpleCylinder(s,trainingConfiguration,simNo):
    maxSize = trainingConfiguration.maxObstacleSize
    gs = vec3(trainingConfiguration.resX, trainingConfiguration.resY, trainingConfiguration.resY)
    NumObsPosX = trainingConfiguration.NumObsPosX
    NumObsPosY = trainingConfiguration.NumObsPosY
    pos = [(simNo % NumObsPosX)*1.0/NumObsPosX, (simNo//NumObsPosX)*1.0/NumObsPosY,0.5]
    obstacle = Cylinder(parent=s, center=gs * vec3(0.21 + 0.79 * pos[0], pos[1], pos[2]), radius=trainingConfiguration.resY * 0.2,
                            z=gs * vec3(0, 0, 1.0))
    return [obstacle]



