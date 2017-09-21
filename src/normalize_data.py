import numpy as np
import os

if __name__ == "__main__":
    files = os.listdir("../res/karman_data")
    scale_factors = []
    for file in files:
        path = os.path.join("../res/karman_data", file)
        data = np.load(path)
        max = np.max(np.abs(data))
        data /= max
        np.save("../res/karman_data_norm/{}".format(file), data)
        scale_factors.append(max)
    print(scale_factors)
    np.save("../res/karman_data_norm/scale_factors", scale_factors)
