import tensorflow as tf
import numpy as np
from subprocess import call

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.truncated_normal(shape, stddev=1)
    return tf.Variable(initial)


def to_image_form(data):
    d = np.reshape(data, (32, 64, 2))
    return np.append(d, np.zeros([32, 64, 1]), axis=2)


def create_net(x):
    hidden_layer_size = 20

    # hidden layer
    w_fc1 = weight_variable([1, hidden_layer_size])
    b_fc1 = bias_variable([hidden_layer_size])
    h_fc1 = tf.nn.sigmoid(tf.matmul(x, w_fc1) + b_fc1)

    # output layer
    w_fc2 = weight_variable([hidden_layer_size, 4096])
    b_fc2 = bias_variable([4096])
    output = tf.tanh(tf.matmul(h_fc1, w_fc2) + b_fc2)
    return output


def create_trainer(output, ground_truth):
    loss = tf.reduce_mean(tf.reduce_sum(tf.pow(ground_truth - output, 2), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(0.1).minimize(loss)
    return train_step, loss


def load_data():
    training_image = []
    training_val = []
    for i in range(31):
        if i == 15: continue
        path = "../res/karman_data_norm/vel" + str(i + 1) + ".npy"
        training_image.append(np.load(path).flatten())
        training_val.append((i + 1) / 32)

    training_data = [np.reshape(training_val, (30, 1)), training_image]
    return training_data


def get_scale_factor(y_pos):
    index = int(y_pos * 32 - 1)
    return np.load("../res/karman_data_norm/scale_factors.npy")[index]


def train():
    x = tf.placeholder(tf.float32, [None, 1])
    output = create_net(x)
    ground_truth = tf.placeholder(tf.float32, [None, 4096])
    train_step, loss = create_trainer(output, ground_truth)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    training_data = load_data()
    # train
    for i in range(5000):
        _, loss_val = sess.run([train_step, loss], feed_dict={x: training_data[0], ground_truth: training_data[1]})
        print("Epoch {}: Loss = {}".format(i, loss_val))

    test_input = 0.5
    net_data = sess.run(output, feed_dict={x: np.reshape([test_input], (1, 1))})
    net_data *= get_scale_factor(test_input)
    net_data += np.load("../res/karman_data_norm/mean.npy").flatten()
    output_img = to_image_form(net_data)
    np.save("../res/net_image", output_img)

    call(["python", "plot_flow.py"])

if __name__ == "__main__":
    train()
