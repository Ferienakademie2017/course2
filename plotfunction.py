
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
                    

def plot (data, name):



    xdim = data.shape[0]
    ydim = data.shape[1]
    zdim = data.shape[2]

    split = np.dsplit(data, zdim)
    x_part = split[0];
    y_part = split[1];

    x_part = np.reshape(x_part, (ydim,xdim))
    y_part = np.reshape(y_part, (ydim,xdim)) 
    
    
    X, Y = np.meshgrid(np.arange(0, xdim, 1), np.arange(0, ydim, 1))

    plt.figure(1)
    Q = plt.quiver(X, Y, x_part, y_part, units='width')

    # rename to save to an other location
    plt.savefig(name + '.pdf')



def plot_error(data_vali, data_out, describtion):
    zdim = data_vali.shape[2]

    split = np.dsplit(data_vali, zdim)
    x_part_v = split[0];
    y_part_v = split[1];

    x_part_v = np.reshape(x_part_v, (32,64))
    y_part_v = np.reshape(y_part_v, (32,64)) 


    split_o = np.dsplit(data_out, zdim)
    x_part_o = split_o[0];
    y_part_o = split_o[1];

    x_part_o = np.reshape(x_part_o, (32,64))
    y_part_o = np.reshape(y_part_o, (32,64)) 

    diff_x = np.subtract (x_part_v,x_part_o)
    diff_x = np.square(diff_x)
    diff_y = np.subtract (y_part_v,y_part_o)
    diff_y = np.square(diff_y)
    res = np.add(diff_x, diff_y)
    err = np.sum(res)
    err = err / x_part_v.size

    X, Y = np.meshgrid(np.arange(0, 64, 1), np.arange(0, 32, 1))

    plt.figure(2)
    Q = plt.quiver(X, Y, x_part_v, y_part_v, units='width', color= 'white', linewidth=0.01)
  
    tick = np.arange(0, 6)
    tick = tick*np.amax(res) * 0.2
    fig = plt.imshow(diff_x, origin='lower', cmap=plt.cm.hot, interpolation='nearest')
    plt.colorbar( ticks=tick, label ='sqared error')
    plt.title('Velocity field and squared Error')

    plt.text(0,-7,'white quivers: validation velocity field')
    string = 'average error: ' + '{:f}'.format(err)
    plt.text(0,-10,string)
    plt.text(-5,40,describtion)
 
    plt.savefig(describtion + '.pdf')

def plot_cost(epoch, cost, name):
    plt.figure(3)
    plt.xlabel('epoch')
    plt.ylabel('cost')
    plt.plot(epoch,cost)
    plt.savefig(name)
