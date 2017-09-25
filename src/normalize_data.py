import numpy as np
import os
from scipy import misc
from routines import to_image_form

# TODO: time, vary x position, look at proposal .txt file. Autoencoder and recurrent neural network?

if __name__ == "__main__":
    files = ["vel{}_{}.npy".format(i, j) for i in range(1, 31) for j in range(100)]

    # calculate mean
    data = np.array([np.load("../res/timestep/{}".format(f)) for f in files])
    print(data.shape)
    mean = np.mean(data, axis=0)
    np.save("../res/timestep_norm/mean", mean)

    scale_factors = []
    for file in files:
        path = os.path.join("../res/timestep", file)
        data = np.load(path)
        m = np.max(np.abs(data))
        data -= mean
        data /= m
        np.save("../res/timestep_norm/{}".format(file), data)
        scale_factors.append(m)
        print("File {} factor {}".format(file, m))
    print(scale_factors)
    np.save("../res/timestep_norm/scale_factors", scale_factors)