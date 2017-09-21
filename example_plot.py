#example how to use plot function

from plotfunction import plot # import it! make shure it is in the same directory

import numpy as np


new_data = np.ones((13,15,2)) # generating some data , you should allready have it! dimensons should look as follows (x,y,2)

plot(new_data, 'name') # plot it! it will be saved in the same directory and named after your second argument
