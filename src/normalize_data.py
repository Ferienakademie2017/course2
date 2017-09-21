import numpy as np
import os

if __name__ == "__main__":
    files = ["vel{}.npy".format(i) for i in range(1, 32)]
    scale_factors = []
    for file in files:
        path = os.path.join("../res/karman_data", file)
        data = np.load(path)
        m = np.max(np.abs(data))
        data /= m
        np.save("../res/karman_data_norm/{}".format(file), data)
        scale_factors.append(m)
        print("File {} factor {}".format(file, m))
    print(scale_factors)
    np.save("../res/karman_data_norm/scale_factors", scale_factors)  # TODO: use scaling, subtract mean, andere error function, convolutional, time
