import numpy as np
import scipy.misc as misc
import matplotlib.pyplot as plt
import pickle

def transformToLinear(inputVectField):
    newSize = np.prod((inputVectField.shape))
    outputVect = np.reshape(inputVectField,newSize,order='c')
    return outputVect

def transformToImage(inputVect,fieldSize):
    outputVectField = np.reshape(inputVect,fieldSize,order='c')
    return outputVectField

training_file_raw = np.load(r'\Ferienakademie\course2\trainingData\trainingKarman1.npy')
training_file_raw = training_file_raw.squeeze()
training_file_plt = training_file_raw - np.min(training_file_raw)
training_file_plt = training_file_plt/(np.max(training_file_plt)-np.min(training_file_plt))
imgplot = plt.imshow(training_file_plt)

training_file_mod = training_file_raw[:,:,0:2]
training_file_lin = transformToLinear(training_file_mod)
training_file_image = transformToImage(training_file_lin, (32,64,2))

plt.show()
#training_file = misc.toimage(training_file_raw)
#training_file.show()