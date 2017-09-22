import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt("../res/training_memorize_all.csv", delimiter=",")
plt.plot(*zip(*data))
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.show()