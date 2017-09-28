from src.routines import *


def create_encoder(x, keep_prob):
    input = tf.reshape(x, (-1, 32, 64, 1))
    conv1 = tf.layers.conv2d(
        inputs=input,
        filters=16,
        kernel_size=[5, 5],
        strides=[2, 2],
        padding="same",
        activation=tf.nn.relu
    )
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=16,
        kernel_size=[5, 5],
        strides=[2, 2],
        padding="same",
        activation=tf.nn.relu
    )
    output = tf.layers.dense(
        inputs=conv2,
        units=64,
        activation=tf.nn.relu
    )
    dropout = tf.layers.dropout(
        inputs=output, rate=keep_prob
    )
    return dropout


def create_decoder(e, keep_prob):
    dense1 = tf.layers.dense(
        inputs=e,
        units=16*8*16,
        activation=tf.nn.relu
    )
    dropout = tf.layers.dropout(
        inputs=dense1, rate=keep_prob
    )
    reshaped1 = tf.reshape(dropout, [-1, 16, 8, 16])
    deconv1 = tf.layers.conv2d_transpose(
        inputs=reshaped1,
        filters=16,
        kernel_size=[5, 5],
        strides=[2, 2],
        padding="same",
        activation=tf.nn.relu
    )
    deconv2 = tf.layers.conv2d_transpose(
        inputs=deconv1,
        filters=16,
        kernel_size=[5, 5],
        strides=[2, 2],
        padding="same",
        activation=tf.nn.relu
    )
    output = tf.layers.dense(
        inputs=deconv2,
        units=32*64,
        activation=tf.nn.tanh
    )
    dropout_output = tf.layers.dropout(
        inputs=output, rate=keep_prob
    )
    return tf.reshape(dropout_output, (-1, 32, 64, 1))


def create_trainer(output, ground_truth):
    loss = tf.reduce_mean(tf.reduce_sum(tf.pow(ground_truth - output, 2), reduction_indices=[1]))
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.piecewise_constant(global_step, [20000, 25000], [0.1, 0.05, 0.01])
    train_step = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)
    return train_step, loss


def create_concatenated_net(x, keep_prob):
    return create_decoder(create_encoder(x, keep_prob), keep_prob)


def train():
    x = tf.placeholder(tf.float32, [None, 4096])
    ground_truth = tf.placeholder(tf.float32, [None, 4096])
    keep_prob = tf.placeholder(tf.float32)
    output = create_concatenated_net(x, keep_prob)
    train_step, loss = create_trainer(output, tf.reshape(output, (-1, 32, 64, 1)))

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    training_data = load_data()

    loss_data = {}
    counter = 0

    # train
    for i in range(50):
        _, loss_val = sess.run([train_step, loss],
                               feed_dict={x: training_data[1][:1], ground_truth: training_data[1][:1], keep_prob: 0.5})
        counter += 1
        print("Loss: {}".format(loss_val))
        if j == len(batch[0]) - 1:
            print("Epoch {}: Loss = {}".format(i, loss_val))
            loss_data[i] = loss_val
            save_csv(loss_data, "../res/autoencoder.csv")

            # test_input = np.load("../res/timestep_norm/vel16_9.npy").flatten()
            # net_data = sess.run(output, feed_dict={x: test_input.reshape(1, 4096), keep_prob: 1.0})
            # net_data *= get_time_scale_factor((0.5, 9))
            # net_data += np.load("../res/timestep_norm/mean.npy").flatten()
            # output_img = to_image_form(net_data)
            #
            # plot(np.load("../res/timestep/vel16_9.npy"), output_img)


if __name__ == "__main__":
    train()
