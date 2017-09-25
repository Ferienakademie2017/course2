import tensorflow as tf
import numpy as np

import Sim1Result
import TrainingConfiguration
import evaluation
import utils

# Discriminator:
# Input: Velocity map: [64, 32], obstacle map: [64, 32] (0: blocked, 1: free)
# Output: [0, 1]; 1 = original data, 0 = generated data

# Generator:
# Input: Obstacle map: [64, 32], noise
# Output: Velocity map: [64, 32]

trainConfig = utils.deserialize("data/test_timeStep/trainConfig.p")
data = trainConfig.loadGeneratedData()
dataPartition = evaluation.DataPartition(len(data), 0.6, 0.4)
utils.serialize(trainConfig.simPath + "dataPartition.p", dataPartition)

trainingData, validationData, testData = dataPartition.computeData(data, exampleType=evaluation.TimeStepSimulationCollection, slice=[0, 1], scale=1)

mb_pos = 0
"""Get the next obstacle and velocity field and noise."""
def get_next_batch(mb_size):
    global mb_pos
    res = [(data.flagField, data.velFields[mb_pos]) for data in trainingData[:mb_size]]
    mb_pos = (mb_pos + 1) % len(trainingData[0].velFields)
    return (*zip(*res), np.random.normal(size=(mb_size, 1)))

# mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
mb_size = 64
obs_dim = [64, 32]
vel_dim = [64, 32, 2]
noise_dim = 1


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

obs_in = tf.placeholder(tf.float32, shape=[None] + obs_dim)
vel_in = tf.placeholder(tf.float32, shape=[None] + vel_dim)
noise_in = tf.placeholder(tf.float32, shape=[None, 1])

""" Discriminator Net model """
# X = tf.placeholder(tf.float32, shape=[None, 784])
# y = tf.placeholder(tf.float32, shape=[None, y_dim])

# D_W1 = tf.Variable(xavier_init([X_dim + y_dim, h_dim]))
# D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

# D_W2 = tf.Variable(xavier_init([h_dim, 1]))
# D_b2 = tf.Variable(tf.zeros(shape=[1]))

# theta_D = [D_W1, D_W2, D_b1, D_b2]

def discriminator(obs_in, vel_in, trainable=True):
    # input = tf.stack([vel_in, obs_in])
    input = vel_in

    size = vel_dim[0]
    for v in vel_dim[1:]:
        size *= v

    layer = tf.reshape(input, [-1, size])
    layer = tf.layers.dense(layer, 256, activation=tf.nn.sigmoid, name="first", trainable=trainable)
    # layer = tf.layers.dense(layer, 64, activation=tf.nn.sigmoid, name="second", trainable=trainable)
    layer = tf.layers.dense(layer, 16, activation=tf.nn.sigmoid, name="third", trainable=trainable)
    layer = tf.nn.sigmoid(layer)
    return layer


    inputs = tf.concat(axis=1, values=[x, y])
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit


""" Generator Net model """
def generator(obs_in, noise_in):
    # input = tf.stack([obs_dim, noise_in], axis=1)
    input = obs_in

    layer = tf.layers.dense(input, 128, activation=tf.nn.sigmoid, name="first")
    layer = tf.layers.dense(layer, 256, name="second")
    output = tf.reshape(layer, [-1] + vel_dim)
    return output

    z = tf.placeholder(tf.float32, shape=[None, Z_dim])

    G_W1 = tf.Variable(xavier_init([Z_dim + y_dim, h_dim]))
    G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    G_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
    G_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

    inputs = tf.concat(axis=1, values=[z, y])
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

with tf.variable_scope("gen"):
    G_sample = generator(obs_in, noise_in)
with tf.variable_scope("disc"):
    D_real = discriminator(obs_in, vel_in)
with tf.variable_scope("disc", reuse=True):
    D_fake = discriminator(obs_in, G_sample, False)

t_vars = tf.trainable_variables()
theta_G = [var for var in t_vars if var.name.startswith("gen/")]
theta_D = [var for var in t_vars if var.name.startswith("disc/")]

D_loss_real = tf.reduce_mean(D_real)
D_loss_fake = tf.reduce_mean(D_fake)
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(D_fake)

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

utils.ensureDir('gan/images')

for i in range(201):
    if i % 100 == 0:
        # Generate validation image
        gen = i // 100
        data = validationData[gen]
        obs = data.flagField
        vel = data.velFields[0]
        noise = np.random.normal(size=(1, 1))
        samples, loss = sess.run([G_sample, G_loss], feed_dict={obs_in: [obs], noise_in: noise})

        print('Validation G loss: {:.4}'.format(loss))
        orig = Sim1Result.Sim1Result(vel, [0], obs, 0)
        res = Sim1Result.Sim1Result(samples[0], [0], obs, 0)
        utils.sim1resToImage(orig, folder="gan/images/")
        utils.sim1resToImage(res, folder="gan/images/")
        utils.sim1resToImage(res, folder="gan/images/", background='error', origRes=orig)

    obs, vel, noise = get_next_batch(mb_size)

    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={obs_in: obs, vel_in: vel})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={obs_in: obs, noise_in: noise})

    if i % 100 == 0:
        print('Iter: {}'.format(i))
        print('D loss: {:.4}'. format(D_loss_curr))
        print('G loss: {:.4}'.format(G_loss_curr))
        print()

# Save variables
utils.ensureDir('gan/training')
name = "gan/training/{}.ckpt".format("final")
saver = tf.train.Saver()
saver.save(sess, name)
