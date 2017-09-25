import tensorflow as tf
from src.routines import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

_, ax = plt.subplots(figsize=(64 / 8, 32 / 8)) # TODO: GIT, transfer learning and continue (save CSV!!!); increase number of time steps with transfer learning

ax.tick_params(
    axis='x',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    bottom='off',  # ticks along the bottom edge are off
    top='off',  # ticks along the top edge are off
    labelbottom='off')
ax.tick_params(
    axis='y',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    left='off',  # ticks along the bottom edge are off
    top='off',  # ticks along the top edge are off
    labelleft='off')


def create_and_save_figure(net_flow, path):
    net_flow = net_flow.transpose(1, 0, 2)
    image_size = net_flow.shape

    skip = 2
    X, Y = np.mgrid[0:image_size[0]:skip, 0:image_size[1]:skip]

    ax.quiver(X, Y, net_flow[::skip, ::skip, 0], net_flow[::skip, ::skip, 1], units='inches')

    plt.savefig(path)
    ax.cla()
    # # plt.show()
    # image = np.append(net_flow, np.zeros([32, 64, 1]), axis=2)
    # misc.imsave(path, image, "png")

if __name__ == "__main__":
    sess = tf.Session()
    saver = tf.train.import_meta_graph("../models/new_model/model-57890.meta")
    saver.restore(sess, tf.train.latest_checkpoint("../models/new_model"))
    output = tf.get_default_graph().get_tensor_by_name("output:0")
    x = tf.get_default_graph().get_tensor_by_name("x:0")

    mean = np.load("../res/timestep_norm/mean.npy").flatten()

    for i in range(100):
        test_input = (5 / 32, i)
        net_data = sess.run(output, feed_dict={x: np.reshape([test_input], (1, 2))})
        net_data *= get_time_scale_factor(test_input)
        net_data += mean
        create_and_save_figure(to_image_form(np.load("../res/timestep/vel5_{}.npy".format(i))), "../res/time_viz_real/{}.png".format(i))
        print("{}%".format(i*100/100))