import subprocess
import numpy as np

sample_set_size = 100
y_position_min = 0.1
y_position_max = 0.9
step_size = (y_position_max - y_position_min) / float(sample_set_size)
y_position_array = np.empty(shape=(sample_set_size))

for y_index in range(sample_set_size):

	y_position = y_position_min + (y_index * step_size)
	y_position_array[y_index] = y_position
	subprocess.call(["manta", "data_generation.py", str(y_position), str(y_index)])

np.save("fluidSamples/y_position_array", y_position_array)
print("Finished fluid data generation of " + str(sample_set_size) + " files")