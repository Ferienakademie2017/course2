
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
                    

def plot (data, name):


    if(data.shape[2] != 2):
        print ('non supported data format! Visit Milena for fixin it!')

    xdim = data.shape[0]
    ydim = data.shape[1]

    split = np.dsplit(data, 2)
    x_part = split[0];
    y_part = split[1];

    x_part = np.reshape(x_part, (xdim,ydim))
    y_part = np.reshape(y_part, (xdim,ydim)) 
    
    
    X, Y = np.meshgrid(np.arange(0, xdim, 1), np.arange(0, ydim, 1))

    plt.figure()
    Q = plt.quiver(X, Y, x_part, y_part, units='width')

    # rename to save to an other location
    plt.savefig(name + '.pdf')




