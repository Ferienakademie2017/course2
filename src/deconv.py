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

    #tensor_var = weight_variable([16,8,4])
    #hidden = tf.multiply(x,tensor_var)

    batch_size = 30
    print("batch size: ", batch_size)
    # fully connected layer
    fc_neurons = 512
    w_fc0 = weight_variable([1,fc_neurons])
    b_fc0 = bias_variable([1,fc_neurons])
    hidden = tf.sigmoid(tf.add(tf.matmul(x,w_fc0),b_fc0))
    hidden = tf.reshape(hidden, [batch_size,16,8,4])
    #hidden: 16x8x4 = 512, 4 channels of 16x8 layers

    # first deconv layer
    batch_size = 30
    output_shape = [batch_size, 32, 16, 4] #[batch_size, height, width, channels]
    strides = [1, 2, 2, 1]
    w1 = tf.get_variable('w1', [5, 5, output_shape[-1], hidden.get_shape()[-1]],
        initializer=tf.random_normal_initializer(stddev=0.1))
    #5x5 filter
    deconv1 = tf.nn.conv2d_transpose(hidden, w1, output_shape=output_shape, strides=strides)

    # second deconv layer
    output_shape = [batch_size, 64, 32, 2] #[batch_size, height, width, channels]
    strides = [1, 2, 2, 1]
    w2 = tf.get_variable('w2', [5, 5, output_shape[-1], deconv1.get_shape()[-1]],
        initializer=tf.random_normal_initializer(stddev=0.1))
    #5x5 filter
    deconv2 = tf.nn.conv2d_transpose(deconv1, w2, output_shape=output_shape, strides=strides)

    #reshape output
    output = tf.reshape(deconv2, [batch_size,4096])
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
        path = "../res/karman_data/vel"+str(i+1)+".npy"
        #current_image = np.reshape(np.load(path).flatten(),[4096])
        current_image = np.load(path).flatten()
        training_image.append(current_image)
        training_val.append((i+1)/32)
    
    training_data = [np.reshape(training_val, (30, 1)),training_image]
    return training_data

def train():
    x = tf.placeholder(tf.float32, [None, 1])
    batch_size = tf.placeholder(tf.int32)
    output = create_net(x)
    ground_truth = tf.placeholder(tf.float32, [None, 4096])
    train_step, loss = create_trainer(output, ground_truth)


    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    
    training_data = load_data()

    train
    for i in range(10000):
        _, loss_val = sess.run([train_step, loss], feed_dict={x: training_data[0], ground_truth: training_data[1]})
        print("Epoch {}: Loss = {}".format(i, loss_val))
    
    #write output
    test_input = np.reshape(0.5*np.ones(30),[30,1])
    #net_output = sess.run(output, feed_dict={x: np.expand_dims(np.array([0.5]), 1)})
    net_output = sess.run(output, feed_dict={x: test_input})
    print(net_output[0].shape)
    output_image = to_image_form(net_output[0])
    np.save("../res/net_image",output_image)



    #saver = tf.train.Saver()
    #saver = tf.train.Saver()
    #saver.save(sess, 'my-model')


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
