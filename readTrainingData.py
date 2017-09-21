import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import pickle

def transformToLinear(inputVectField):
    newSize = np.prod((inputVectField.shape))
    outputVect = np.reshape(inputVectField,newSize,order='c')
    return outputVect

def transformToImage(inputVect,fieldSize):
    outputVectField = np.reshape(inputVect,fieldSize,order='c')
    return outputVectField

def scaleImage(inputField,scaleParam):
    scaledField = ndimage.zoom(inputField, [1, 1/scaleParam, 1/scaleParam, 1])
    return scaledField

def showImage(inputField):
    scaledImage = inputField - np.min(inputField)
    scaledImage = scaledImage / (np.max(scaledImage) - np.min(scaledImage))
    imgplot = plt.imshow(scaledImage)
    plt.show()

def loadData(path):
    sourceFile = open(path, "rb")
    yPositions = pickle.load(sourceFile)
    trainingSize = yPositions.shape[0]
    # fieldSize = (32, 64, 2)
    outputSize = 256

    trainingInput = yPositions
    trainingOutput = np.zeros((trainingSize, outputSize), np.float32)

    for i in range(trainingSize):
        currentOutput = pickle.load(sourceFile)
        # showVectorField(currentOutput[0,:,:,:])
        currentOutput = scaleImage(currentOutput, 2)
        currentOutput = transformToLinear(currentOutput[0, :, :, 0:2])
        trainingOutput[i, :] = currentOutput

    return trainingInput,trainingOutput

# def showVectorField(inputField):
#     fig = plt.figure()
#     M = np.hypot(inputField[:,:,0], inputField[:,:,1])
#     [baseY,baseX] = np.meshgrid(range(inputField.shape[0]),range(inputField.shape[1]))
#     plt.quiver(baseX, baseY, inputField[:,:,1], inputField[:,:,0], M)
#     plt.show()

if __name__ == "__main__":
    sourceFile = open( r'C:\Users\Nico\Documents\Ferienakademie\course2\trainingData\trainingKarman1.p', "rb" )
    yPositions = pickle.load(sourceFile)
    trainingSize = yPositions.shape[0]
    fieldSize = (32,64,2)
    outputSize = 256

    trainingInput = yPositions
    trainingOutput = np.zeros((trainingSize,outputSize),np.float32)

    for i in range(trainingSize):
        currentOutput = pickle.load(sourceFile)
        # showVectorField(currentOutput[0,:,:,:])
        currentOutput = scaleImage(currentOutput, 2)
        currentOutput = transformToLinear(currentOutput[0,:,:,0:2])
        trainingOutput[i,:] = currentOutput