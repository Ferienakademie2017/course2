import tensorflow as tf
import numpy as np
import time

import Sim1Result
import TrainingConfiguration
import evaluation
import utils

# Discriminator:
# Input: Velocity map: [64, 32, 2], obstacle map: [64, 32] (0: blocked, 1: free)
# Output: [0, 1]; 1 = original data, 0 = generated data

# Generator:
# Input: Obstacle map: [64, 32], noise
# Output: Velocity map: [64, 32, 2]

time_start = time.perf_counter()
time_check = time_start

def log_time(name):
    global time_check
    cur = time.perf_counter()
    print("{:.4}s  {}  ({:.4}s elapsed)".format(cur - time_start, name, cur - time_check))
    time_check = cur

trainConfig = utils.deserialize("data/test_timeStep/trainConfig.p")
data = trainConfig.loadGeneratedData()
log_time("Load data")
dataPartition = evaluation.DataPartition(len(data), 0.6, 0.4)
utils.serialize(trainConfig.simPath + "dataPartition.p", dataPartition)

trainingData, validationData, testData = dataPartition.computeData(data, exampleType=evaluation.TimeStepSimulationCollection, slice=[0, 1], scale=1)
log_time("Partition data")

mb_pos = 0
"""Get the next obstacle and velocity field and noise."""
def get_next_batch(mb_size):
    global mb_pos
    res = [(data.flagField, data.velFields[mb_pos]) for data in trainingData[:mb_size]]
    mb_pos = (mb_pos + 1) % len(trainingData[0].velFields)
    return (*zip(*res), np.random.normal(size=(len(res), 1)))

mb_size = 16
obs_dim = [64, 32]
vel_dim = [64, 32, 2]
noise_dim = 1


obs_in = tf.placeholder(tf.float32, shape=[None] + obs_dim)
vel_in = tf.placeholder(tf.float32, shape=[None] + vel_dim)
noise_in = tf.placeholder(tf.float32, shape=[None, 1])

def count_vars():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters
    print("Total variables:", total_parameters)

""" Discriminator Net model """
def discriminator(obs_in, vel_in, trainable=True):
    layer = tf.concat([vel_in, tf.expand_dims(obs_in, -1)], 3)

    numFeatures = 2
    kernelSize = [5, 5]
    zoomSteps = 5

    weights_ini = tf.truncated_normal_initializer(stddev=0.02)

    # Zoom out
    for i in range(zoomSteps):
        layer = tf.contrib.layers.conv2d_transpose(layer, numFeatures,
            kernelSize, 2, "SAME", activation_fn=tf.nn.relu,
            weights_initializer=weights_ini, trainable=trainable)
        layer = tf.contrib.layers.batch_norm(layer, decay=0.9,
            updates_collections=None, epsilon=1e-5, scale=True)
            # scope="batch_norm3_{}".format(i))

    size = vel_dim[0]
    for v in vel_dim[1:]:
        size *= v

    size //= 2 ** zoomSteps
    layer = tf.reshape(layer, [-1, size])
    layer = tf.layers.dense(layer, 1, activation=tf.nn.sigmoid, name="last",
        trainable=trainable)

    return layer

""" Generator Net model """
def generator(obs_in, noise_in):
    noise = noise_in
    for _ in range(len(obs_dim)):
        noise = tf.expand_dims(noise, -1)
    noise = tf.tile(noise, [1] + obs_dim + [1])
    obs = tf.expand_dims(obs_in, -1)
    layer = tf.concat([obs, noise], 3)

    numFeatures = 10
    kernelSize = [5, 5]
    zoomSteps = 3
    zoomLayers = []

    weights_ini = tf.truncated_normal_initializer(stddev=0.02)

    # Zoom in
    for i in range(zoomSteps):
        zoomLayers.append(layer)
        layer = tf.contrib.layers.conv2d(layer, numFeatures,
            kernelSize, 2, "SAME", activation_fn=tf.nn.relu,
            weights_initializer=weights_ini)
        # layer = tf.contrib.layers.batch_norm(layer, decay=0.9,
            # updates_collections=None, epsilon=1e-5, scale=True,
            # scope="batch_norm0_{}".format(i))

    # Zoom out
    for i in range(zoomSteps):
        layer = tf.contrib.layers.conv2d_transpose(layer, numFeatures,
            kernelSize, 2, "SAME", activation_fn=tf.nn.relu,
            weights_initializer=weights_ini)
        # layer = tf.contrib.layers.batch_norm(layer, decay=0.9,
            # updates_collections=None, epsilon=1e-5, scale=True,
            # scope="batch_norm3_{}".format(i))
        layer = tf.concat([layer, zoomLayers[zoomSteps - 1 - i]], 3)

        # layer = tf.contrib.layers.conv2d(layer, numFeatures,
            # kernelSize, 1, "SAME", activation_fn=tf.nn.relu,
            # weights_initializer=weights_ini)
        # layer = tf.contrib.layers.batch_norm(layer, decay=0.9,
            # updates_collections=None, epsilon=1e-5, scale=True,
            # scope="batch_norm4_{}".format(i))
        # layer = tf.contrib.layers.conv2d(layer, numFeatures,
            # kernelSize, 1, "SAME", activation_fn=tf.nn.relu,
            # weights_initializer=weights_ini)
        # layer = tf.contrib.layers.batch_norm(layer, decay=0.9,
            # updates_collections=None, epsilon=1e-5, scale=True,
            # scope="batch_norm5_{}".format(i))

    layer = tf.contrib.layers.conv2d(layer, 2, kernelSize,
        1, "SAME", activation_fn=None,
        weights_initializer=weights_ini)

    return layer

with tf.variable_scope("gen"):
    G_sample = generator(obs_in, noise_in)
    log_time("Create generator")
with tf.variable_scope("disc"):
    D_real = discriminator(obs_in, vel_in)
    log_time("Create real discriminator")
with tf.variable_scope("disc", reuse=True):
    D_fake = discriminator(obs_in, G_sample, False)
    log_time("Create fake discriminator")

t_vars = tf.trainable_variables()
theta_G = [var for var in t_vars if var.name.startswith("gen/")]
theta_D = [var for var in t_vars if var.name.startswith("disc/")]

D_loss_real = tf.reduce_mean(D_real)
D_loss_fake = tf.reduce_mean(D_fake)
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(D_fake)

count_vars()

D_solver = tf.train.AdamOptimizer(0.02).minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer(0.02).minimize(G_loss, var_list=theta_G)
log_time("Create optimizers")


sess = tf.Session()
sess.run(tf.global_variables_initializer())
log_time("Init tensorflow")

utils.ensureDir('gan/images')

for i in range(101):
    obs, vel, noise = get_next_batch(mb_size)
    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={obs_in: obs, vel_in: vel, noise_in: noise})
    obs, vel, noise = get_next_batch(mb_size)
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={obs_in: obs, noise_in: noise})

    if i % 10 == 0:
        log_time("10 training steps")
        print('Iter: {}'.format(i))
        print('D loss: {:.4}'. format(D_loss_curr))
        print('G loss: {:.4}'.format(G_loss_curr))

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
        print()

log_time("Finish learning")

# Save variables
utils.ensureDir('gan/training')
name = "gan/training/{}.ckpt".format("final")
saver = tf.train.Saver()
saver.save(sess, name)
log_time("Save data")
