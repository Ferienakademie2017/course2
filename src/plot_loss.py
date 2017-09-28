# import matplotlib.pyplot as plt
# import numpy as np
#
# data = np.loadtxt("../res/training_memorize_all.csv", delimiter=",")
# plt.ylim((0,80))
# plt.plot(*zip(*data))
# plt.ylabel("Loss")
# plt.xlabel("Epoch")
# plt.show()

import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt("../res/autoencoder.csv", delimiter=",")
plt.figure(figsize=(10, 5))
# plt.ylim((0,80))
plt.plot(*zip(*data))
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.show()