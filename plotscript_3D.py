
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt


# In[42]:

def plot (data, name):
# put your np array here -> it is supposed to look like (64 x 32 x 3)
#data = np.load('/home/milena/Ferienakademie/course2/data.npy')
	split = np.dsplit(data, 3)
	x_part = split[0];
	y_part = split[1];

	x_part = np.reshape(x_part, (64,32))
	y_part = np.reshape(y_part, (64,32)) 
		                


	# In[58]:


	X, Y = np.meshgrid(np.arange(0, 64, 1), np.arange(0, 32, 1))

	plt.figure()
	Q = plt.quiver(X, Y, x_part, y_part, units='width')

	# rename to save to an other location
	plt.savefig(name)

