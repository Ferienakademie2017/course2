import numpy as np
import os
from scipy import misc
from src.memorize_all import to_image_form

# TODO: subtract mean, stddev(?), convolutional, time

if __name__ == "__main__":
    files = ["vel{}.npy".format(i) for i in range(1, 32)]

    # calculate mean
    data = np.array([np.load("../res/karman_data/{}".format(f)) for f in files])
    print(data.shape)
    mean = np.mean(data, axis=0)
    np.save("../res/karman_data_norm/mean", mean)

    scale_factors = []
    for file in files:
        path = os.path.join("../res/karman_data", file)
        data = np.load(path)
        m = np.max(np.abs(data))
        data -= mean
        data /= m
        np.save("../res/karman_data_norm/{}".format(file), data)
        scale_factors.append(m)
        print("File {} factor {}".format(file, m))
    print(scale_factors)
    np.save("../res/karman_data_norm/scale_factors", scale_factors)