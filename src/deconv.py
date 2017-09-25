from routines import *

def create_net(x):
    batch_size = 30

    # fully connected layer
    fc_neurons = 1024
    w_fc0 = weight_variable([1,fc_neurons])
    b_fc0 = bias_variable([1,fc_neurons])
    hidden = tf.tanh(tf.add(tf.matmul(x,w_fc0),b_fc0))
    hidden = tf.reshape(hidden, [batch_size,16,8,8]) #hidden: 16x8x4 = 512, 4 channels of 16x8 layers

    # first deconv layer
    output_shape = [batch_size, 32, 16, 4] #[batch_size, height, width, channels]
    strides = [1, 2, 2, 1]
    w1 = weight_variable([5, 5, output_shape[-1],int(hidden.get_shape()[-1])])
    deconv1 = tf.nn.conv2d_transpose(hidden,
                                     w1, 
                                     output_shape=output_shape,
                                     strides=strides)
    #leaky ReLU
    deconv1 = tf.maximum(deconv1, 0.2*deconv1)

    # second deconv layer
    output_shape = [batch_size, 64, 32, 2] #[batch_size, height, width, channels]
    strides = [1, 2, 2, 1]
    w2 = weight_variable([5, 5, output_shape[-1],int(deconv1.get_shape()[-1])])
    deconv2 = tf.nn.conv2d_transpose(deconv1,
                                     w2, 
                                     output_shape=output_shape,
                                     strides=strides)

    output = tf.reshape(deconv2, [batch_size,4096])
    return tf.tanh(output)

def create_trainer(output, ground_truth):
    loss = tf.reduce_mean(tf.reduce_sum(tf.pow(ground_truth - output, 2), reduction_indices=[1]))
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.piecewise_constant(global_step, [10000, 8000], [0.1, 0.05, 0.01])
    train_step = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)
    return train_step, loss

def train():
    x = tf.placeholder(tf.float32, [None, 1])
    output = create_net(x)
    ground_truth = tf.placeholder(tf.float32, [None, 4096])
    train_step, loss = create_trainer(output, ground_truth)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    
    training_data = load_data()

    for i in range(1000):
        _, loss_val = sess.run([train_step, loss], feed_dict={x: training_data[0], ground_truth: training_data[1]})
        print("Epoch {}: Loss = {}".format(i, loss_val))
    
    #write output
    test_input = np.reshape(0.5*np.ones(30),[30,1])
    net_output = sess.run(output, feed_dict={x: test_input})
    net_output[0] *= get_scale_factor(test_input[0])
    net_output[0] += np.load("../res/karman_data_norm/mean.npy").flatten()
    output_image = to_image_form(net_output[0])
    np.save("../res/net_image",output_image)

    plot()

if __name__ == "__main__":
    train()