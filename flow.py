import scipy.misc as misc
import numpy as np
from numpy import linalg as LA
import math
import matplotlib.pyplot as plt

def plot_flow_triple(real_flow, net_flow):
    S = 10  # scaling parameter
    # takes ONE real flow and ONE output from network and compares them
    real_flow = real_flow.transpose((1,0,2))
    net_flow  =  net_flow.transpose((1,0,2))
    image_size = real_flow.shape

    X,Y = np.mgrid[0:image_size[0]:1, 0:image_size[1]:1]

    [f, (ax1, ax2, ax3)] = plt.subplots(3, sharex=True, sharey=True)
    M = np.hypot(real_flow[::1,::1,0], real_flow[::1,::1,1])
    ax1.quiver(X,Y,real_flow[::1,::1,0], real_flow[::1,::1,1],M, scale=S)
    ax1.set_title("Real flow")
    ax1.set_xlim(0, image_size[0])
    ax1.set_ylim(0, image_size[1])
    ax2.set_title("Output of network")
    ax2.quiver(X,Y, net_flow[::1,::1,0], net_flow[::1,::1,1],M, scale=S)

    ax3.set_title("Velocity differences (real-network)")
    N=np.hypot(real_flow[::1,::1,0]-net_flow[::1,::1,0], real_flow[::1,::1,1]-net_flow[::1,::1,1])
    ax3.quiver(X,Y,real_flow[::1,::1,0]-net_flow[::1,::1,0],real_flow[::1,::1,1]-net_flow[::1,::1,1],N, scale=S)

    # compute error
    diff_flow = (real_flow[:,:,0]-net_flow[:,:,0])**2 + (real_flow[:,:,1]-net_flow[:,:,1])**2
    diff_norm = math.sqrt(np.sum(diff_flow))

    real_flow_sq = real_flow[:,:,0]**2 + real_flow[:,:,1]**2
    real_norm = math.sqrt(np.sum(real_flow_sq))
    print("Average error: %f" % (diff_norm/real_norm))

    plt.show()
    exit()
