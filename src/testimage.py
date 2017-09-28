import math
import numpy as np
import scipy

import Sim1Result
import utils

# Test the arrToImage function
data = np.zeros((64, 32, 3), dtype='f')
obs = np.zeros((64, 32), dtype='f')
smoke = np.zeros((64, 32), dtype='f')
mid = (len(obs) // 2, len(obs[0]) // 2)
maxdist = math.sqrt(mid[0] ** 2 + mid[1] ** 2)
for i in range(len(data)):
    for j in range(len(data[i])):
        rel = (i - mid[0], j - mid[1])
        dist = math.sqrt(rel[0] ** 2 + rel[1] ** 2)
        angle = math.atan2(rel[1], rel[0])
        if rel[0] > 0:
            obs[i][j] = 1
            data[i][j][0] = 1
            data[i][j][1] = 0
        elif rel[1] > 0:
            obs[i][j] = 0.5
            data[i][j][0] = 0
            data[i][j][1] = 1
        # print("X: {}, Y: {}, Angle: {}".format(i, j, angle))
        data[i][j][0] = dist * math.sin(-angle) / maxdist * 5
        data[i][j][1] = dist * math.cos(angle) / maxdist * 5
        data[i][j][2] = 0
        smoke[i][j] = i / len(data)
result = Sim1Result.Sim1Result(data, (0, 0, 0), obs, 0)
utils.sim1resToImage(result, smokeField=smoke)
