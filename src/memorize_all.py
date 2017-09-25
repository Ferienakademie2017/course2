from routines import *

def create_net(x): #TODO: change net architecture
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
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.piecewise_constant(global_step, [20000, 25000], [0.1, 0.05, 0.01])
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

    loss_data = {}
    mean =  np.load("../res/karman_data_norm/mean.npy").flatten()

    # train
    for i in range(2000):
        _, loss_val = sess.run([train_step, loss], feed_dict={x: training_data[0], ground_truth: training_data[1]})
        loss_data[i] = loss_val
        if i % 100 == 0:
            print("Epoch {}: Loss = {}".format(i, loss_val))

        test_input = 0.5
        net_data = sess.run(output, feed_dict={x: np.reshape([test_input], (1, 1))})
        net_data *= get_scale_factor(test_input)
        net_data += mean
        output_img = to_image_form(net_data)
        np.save("../res/visualization_data/{}".format(i), output_img)

    #save_csv(loss_data, "../res/training_memorize_all.csv")

    test_input = 0.5
    net_data = sess.run(output, feed_dict={x: np.reshape([test_input], (1, 1))})
    net_data *= get_scale_factor(test_input)
    net_data += np.load("../res/karman_data_norm/mean.npy").flatten()
    output_img = to_image_form(net_data)
    # np.save("../res/net_image", output_img)

    plot()

if __name__ == "__main__":
    train()
