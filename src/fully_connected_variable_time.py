from src.routines import *


def create_net(x):
    sizes = [2, 50, 4096]

    # hidden layer
    w_fc1 = weight_variable(sizes[:2])
    b_fc1 = bias_variable([sizes[1]])
    h_fc1 = leaky_relu(tf.matmul(x, w_fc1) + b_fc1, 0.01)

    # # hidden layer #2
    # w_fc2 = weight_variable(sizes[1:3])
    # b_fc2 = bias_variable([sizes[2]])
    # h_fc2 = leaky_relu(tf.matmul(h_fc1, w_fc2) + b_fc2, 0.1)

    # output layer
    w_fc3 = weight_variable(sizes[1:3])
    b_fc3 = bias_variable([sizes[2]])
    output = tf.tanh(tf.matmul(h_fc1, w_fc3) + b_fc3, name="output")
    return output


def create_trainer(output, ground_truth):
    loss = tf.reduce_mean(tf.reduce_sum(tf.pow(ground_truth - output, 2), reduction_indices=[1]), name="loss")
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.piecewise_constant(global_step, [100, 3000], [0.05, 0.01, 0.005], name="lr") # TODO: learning rate weiter runter machen
    train_step = tf.train.AdamOptimizer(lr, name="AdamOptimizer").minimize(loss, global_step=global_step, name="train_step")
    return train_step, loss


def train():
    x = tf.placeholder(tf.float32, [None, 2], name="x")
    output = create_net(x)
    ground_truth = tf.placeholder(tf.float32, [None, 4096], name="ground_truth")
    train_step, loss = create_trainer(output, ground_truth)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    saver = tf.train.Saver(var_list=[output])
    # saver = tf.train.import_meta_graph("../models/new_model/model-57890.meta")
    saver.restore(sess, tf.train.latest_checkpoint("../models/new_model")) # TODO: create a new saver that only saves the variables I want. Use that for the training etc.

    # graph = tf.get_default_graph()
    # output = graph.get_tensor_by_name("output:0")
    # x = graph.get_tensor_by_name("x:0")
    # ground_truth = graph.get_tensor_by_name("ground_truth:0")
    # train_step, loss = create_trainer(output, ground_truth)

    training_data = load_time_data()

    loss_data = {}
    # mean = np.load("../res/timestep_norm/mean.npy").flatten()

    save = False
    counter = 0

    # train
    for i in range(10000):
        batch = create_mini_batches(training_data, 50)
        for j in range(1, len(batch[0])):
            _, loss_val = sess.run([train_step, loss],
                                            feed_dict={x: batch[0][j], ground_truth: batch[1][j]})
            counter += 1
            if save and loss_val <= 5:
                saver = tf.train.Saver()
                saver.save(sess, '../models/new_model/model', global_step=counter)
                print("Saved a model.")
                save = False
            if j == len(batch[0]) - 1:
                print("Epoch {}: Loss = {}".format(i, loss_val))
                loss_data[i] = loss_val

    # save_csv(loss_data, "../res/timestep.csv")
    saver = tf.train.Saver()
    # saver.save(sess, '../models/new_model/lastmodel', global_step=counter)
    print("Saved last model.")

if __name__ == "__main__":
    train()
