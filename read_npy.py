import numpy as np
import scipy.misc as misc
import matplotlib.pyplot as plt

training_file_raw = np.load(r'C:\Users\Nico\Documents\Ferienakademie\course2\trainingData\trainingKarman1.npy')
training_file_raw = training_file_raw.squeeze()
training_file_plt = training_file_raw - np.min(training_file_raw)
training_file_plt = training_file_plt/(np.max(training_file_plt)-np.min(training_file_plt))
imgplot = plt.imshow(training_file_plt)
plt.show()
#training_file = misc.toimage(training_file_raw)
#training_file.show()