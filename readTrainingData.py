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
    if inputField.shape[2] < 3:
        inputField[:,:,2] = np.zeros(inputField.shape[0:2])
    scaledImage = inputField - np.min(inputField)
    scaledImage = scaledImage / (np.max(scaledImage) - np.min(scaledImage))
    imgplot = plt.imshow(scaledImage)
    plt.show()

def loadData(path, filterLimit = 5):
    sourceFile = open(path, "rb")
    yPositions = pickle.load(sourceFile)
    trainingSize = yPositions.shape[0]
    # fieldSize = (32, 64, 2)
    outputSize = 256
    filterCounter = 0

    # trainingInput = yPositions
    trainingInput = []
    trainingOutput = np.zeros((trainingSize, outputSize), np.float32)

    for i in range(trainingSize):
        currentOutput = pickle.load(sourceFile)
        if np.max(np.abs(currentOutput)) > filterLimit:
            filterCounter += 1
            continue
        # showVectorField(currentOutput[0,:,:,:])
        currentOutput = scaleImage(currentOutput, 2)
        currentOutput = transformToLinear(currentOutput[0, :, :, 0:2])
        trainingInput.append(yPositions[i])
        trainingOutput[i-filterCounter, :] = currentOutput
    print("droped ", filterCounter, ' of ',trainingOutput.shape[0], " samples from training set")
    trainingOutput = trainingOutput[0:(trainingOutput.shape[0]-filterCounter),:]
    trainingInput = np.array(trainingInput)

    return trainingInput,trainingOutput

def loadDataTimeSequence(path, filterLimit = 5, loadPrefiltered = False, savePrefiltered = False):
    savePath = path + 'f'
    if loadPrefiltered:
        sourceFile = open(savePath, "rb")
        trainingInput = pickle.load(sourceFile)
        trainingOutput = pickle.load(sourceFile)
        return trainingInput, trainingOutput

    sourceFile = open(path, "rb")
    yPositions = pickle.load(sourceFile)
    trainingSize = yPositions.shape[0]
    # fieldSize = (32, 64, 2)
    height = 8
    width = 16
    timesteps = 50
    outputSize = 256
    filterCounter = 0

    # trainingInput = yPositions
    trainingInput = []
    trainingOutput = np.zeros((trainingSize, height, width, 2*timesteps), np.float32)

    for i in range(trainingSize):
        currentOutput = pickle.load(sourceFile)
        if np.max(np.abs(currentOutput)) > filterLimit:
            filterCounter += 1
            continue
        # showVectorField(currentOutput[0,:,:,:])
        trainingInput.append(yPositions[i])
        for k in range(len(currentOutput)):
            currentOutput[k] = scaleImage(currentOutput[k], 2)
            trainingOutput[i-filterCounter, :,:,2*k:2*(k+1)] = currentOutput[k][0,:,:,0:2]
    print("droped ", filterCounter, ' of ',trainingOutput.shape[0], " samples from training set")
    trainingOutput = trainingOutput[0:(trainingOutput.shape[0]-filterCounter),:,:,:]
    trainingInput = np.array(trainingInput)

    if savePrefiltered:
        targetFile = open(savePath, "wb")
        pickle.dump(trainingInput,targetFile)
        pickle.dump(trainingOutput,targetFile)

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
        showImage(currentOutput[0,:,:,:])
        currentOutput = scaleImage(currentOutput, 2)
        currentOutput = transformToLinear(currentOutput[0,:,:,0:2])
        trainingOutput[i,:] = currentOutput