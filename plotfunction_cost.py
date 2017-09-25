import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
   

def plot_cost(epoch, cost, name):
	plt.xlabel('epoch')
	plt.ylabel('cost')
	plt.plot(epoch,cost)
	plt.savefig(name)
