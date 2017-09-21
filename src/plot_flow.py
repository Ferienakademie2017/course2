import scipy.misc as misc
import numpy as np

import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    real_flow = np.load("../res/karman_data/vel16.npy")
    net_flow = np.load("../res/net_image.npy")
    
    # takes ONE output and produces a quiver plot from it
    real_flow = real_flow.transpose((1,0,2))
    net_flow  =  net_flow.transpose((1,0,2))
    image_size = real_flow.shape
    
    # quiver version
    X,Y = np.mgrid[0:image_size[0]:2, 0:image_size[1]:2]
    
    [f, (ax1, ax2)] = plt.subplots(2, sharex=True, sharey=True)
    ax1.quiver(X,Y,real_flow[::2,::2,0],real_flow[::2,::2,1])
    ax1.set_title("Real flow")
    ax1.set_xlim(0, image_size[0])
    ax1.set_ylim(0, image_size[1])
    ax2.set_title("Output of network")
    ax2.quiver(X,Y, net_flow[::2,::2,0], net_flow[::2,::2,1])
    
    plt.show()
    exit()
    