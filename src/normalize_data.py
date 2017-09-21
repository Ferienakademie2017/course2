import numpy as np
import os

if __name__ == "__main__":
    files = os.listdir("../res/karman_data")
    for file in files:
        path = os.path.join("../res/karman_data", file)
        data = np.load(path)
        # scale to [-1, 1]
        data = 2 * (data - np.max(data)) / -np.ptp(data) - 1
        np.save("../res/karman_data_norm/{}".format(file), data)