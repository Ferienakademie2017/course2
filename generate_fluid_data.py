import subprocess
import numpy as np
import shutil
import os

output_folder_6432 = "fluidSamples6432/"
output_folder_6432_images = "fluidSamples6432Images/"
output_folder_1608 = "fluidSamples1608/"
output_folder_metadata = "fluidSamplesMetadata/"


y_position_min = 2
y_position_max = 30
possible_positions = y_position_max - y_position_min
step_size = 1

# how many times to simulate through all possible positions
iterations = 1
y_position_array = np.empty(shape=(iterations*possible_positions))


def clear_output_folders(path):
	if not os.path.exists(path):
		os.makedirs(path)
	# delete output folder content
	for file in os.listdir(path):
		file_path = os.path.join(path, file)
		try:
			if os.path.isfile(file_path):
				os.unlink(file_path)
		except Exception as e:
			print(e)

clear_output_folders(output_folder_6432)
clear_output_folders(output_folder_6432_images)
clear_output_folders(output_folder_1608)
clear_output_folders(output_folder_metadata)

for iteration in range(iterations):
	for y_index in range(possible_positions):
		y_position = y_position_min + (y_index * step_size)
		y_position_array[y_index] = y_position
		subprocess.call(["manta", "data_generation.py", str(y_position), str((iteration*100) + y_index)])

# save position array
np.save(output_folder_metadata + "y_position_array", y_position_array)
print("Finished fluid data generation of " + str(iterations * possible_positions) + " files")
