import tensorflow as tf
import scipy.misc as misc
import numpy as np


def weight_variable(shape):
    initial = tf.truncated_normal(shape)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.truncated_normal(shape)
    return tf.Variable(initial)


def to_image_form(data):
    d = np.reshape(data, (32, 64, 2))
    return np.append(d, np.zeros([32, 64, 1]), axis=2)


def create_net(x):
    # output layer
    w_fc1 = weight_variable([1, 4096])
    b_fc1 = bias_variable([4096])
    output = tf.matmul(x, w_fc1) + b_fc1
    return output


def create_trainer(output, ground_truth):
    loss = tf.reduce_mean(tf.reduce_sum(tf.pow(ground_truth - output, 2), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    return train_step, loss

def load_data():
    training_image = []
    training_val = []
    for i in range(31):
        path = "../res/karman_data/vel"+str(i+1)+".npy"
        training_image.append(np.load(path).flatten())
        training_val.append((i+1)/32)
    
    training_data = [np.reshape(training_val, (31, 1)),training_image]
    return training_data

def train():
    x = tf.placeholder(tf.float32, [None, 1])
    output = create_net(x)
    ground_truth = tf.placeholder(tf.float32, [None, 4096])
    train_step, loss = create_trainer(output, ground_truth)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    
    training_data = load_data()

    #train
    for i in range(120):
        #np.expand_dims(np.array([0.5]), 1
        #np.expand_dims(data, 1).transpose()
        _, loss_val = sess.run([train_step, loss], feed_dict={x: training_data[0], ground_truth: training_data[1]})
        print("Epoch {}: Loss = {}".format(i, loss_val))

    # data = to_image_form(data)
    # net_data = sess.run(output, feed_dict={x: np.expand_dims(np.array([0.5]), 1)})


if __name__ == "__main__":
    train()

    # misc.imsave("/home/mathias/PycharmProjects/Ferienakademie/original.png", data, "png")
    # misc.imsave("/home/mathias/PycharmProjects/Ferienakademie/net.png", net_data, "png")
    # misc.imsave("/home/mathias/PycharmProjects/Ferienakademie/diff.png", diff, "png")

    # epochs = 30
    # mini_batch_size = 10
    # mnist_train_size = 10000
    # for i in range(epochs):
    #     mini_batches = [mnist.train.next_batch(mini_batch_size) for i in range(int(mnist_train_size / mini_batch_size))]
    #     for mini_batch in mini_batches:
    #         sess.run(train_step, feed_dict={x: mini_batch[0], y_: mini_batch[1]})
    #     print("Epoch {} accuracy: {:.2f}%".format(i+1, get_accuracy() * 100))
    #
    # print("Final accuracy: {:.2f}%".format(get_accuracy() * 100))
