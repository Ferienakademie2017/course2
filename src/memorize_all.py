import tensorflow as tf
import scipy.misc as misc
import numpy as np
import matplotlib.pyplot as plt

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev = 0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.truncated_normal(shape,stddev = 0.1)
    return tf.Variable(initial)


def to_image_form(data):
    d = np.reshape(data, (32, 64, 2))
    return np.append(d, np.zeros([32, 64, 1]), axis=2)



def create_net(x):
    # hidden layer
    hidden_neurons = 20
    w_fc0 = weight_variable([1,hidden_neurons])
    b_fc0 = bias_variable([1,hidden_neurons])
    hidden = tf.sigmoid(tf.add(tf.matmul(x,w_fc0),b_fc0))

    # output layer
    w_fc1 = weight_variable([hidden_neurons, 4096])
    b_fc1 = bias_variable([4096])
    #output = tf.multiply(10.,tf.tanh(tf.add(tf.matmul(hidden, w_fc1),b_fc1)))
    output = tf.tanh(tf.add(tf.matmul(hidden, w_fc1),b_fc1))
    return output


def create_trainer(output, ground_truth):
    loss = tf.reduce_mean(tf.reduce_sum(tf.pow(ground_truth - output, 2), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(0.1).minimize(loss)
    return train_step, loss

def load_data():
    training_image = []
    training_val = []
    for i in range(31):
        if i==15: continue
        path = "../res/karman_data_norm/vel"+str(i+1)+".npy"
        training_image.append(np.load(path).flatten())
        training_val.append((i+1)/32)
    
    training_data = [np.reshape(training_val, (30, 1)),training_image]
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
    for i in range(2000):
        _, loss_val = sess.run([train_step, loss], feed_dict={x: training_data[0], ground_truth: training_data[1]})
        print("Epoch {}: Loss = {}".format(i, loss_val))


    saver = tf.train.Saver()
    saver = tf.train.Saver()
    saver.save(sess, 'my-model')

    #write output
    net_output = sess.run(output, feed_dict={x: np.expand_dims(np.array([0.5]), 1)})
    output_image = to_image_form(net_output)
    #output_image = output_image.transpose((1,0,2))
    np.save("../res/net_image",output_image)

if __name__ == "__main__":
    train()

    #sess = tf.Session()
    #new_saver = tf.train.import_meta_graph('my-model.meta')
    #new_saver.restore(sess, tf.train.latest_checkpoint('./'))
    #net_output = sess.run(output, feed_dict={x: np.expand_dims(np.array([0.5]), 1)})

   
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
