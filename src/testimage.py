import math
import numpy as np
import utils
import scipy

# Test the arrToImage function
data = np.zeros((64, 32, 2), dtype='f')
mid = (32, 16)
maxdist = math.sqrt(mid[0] ** 2 + mid[1] ** 2)
for i in range(len(data)):
	for j in range(len(data[i])):
		rel = (i - mid[0], j - mid[1])
		dist = math.sqrt(rel[0] ** 2 + rel[1] ** 2)
		angle = math.atan2(rel[0], rel[1])
		data[i][j][0] = dist * math.cos(-angle) / maxdist
		data[i][j][1] = dist * math.sin(-angle) / maxdist
utils.arrToImage(data)
