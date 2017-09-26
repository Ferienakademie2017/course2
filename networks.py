"""Factory methods to create networks (i.e. the input and the output layer).
Input: single scalar. Output: flat array, dimensions derived from 'data'
argument."""

import tensorflow as tf


def create_1_8_256_network(data):
    """NETWORK ARCHITECTURE:
       Input        Layer1          Output-Layer
       1 (lin)  ->  8 (tanh)   ->   X*Y (lin)
       X, Y are the data dimensions
    """
    layer1_size = 8

    # derive output size from size of single data sample
    output_size = 1
    for d in range(1, data.ndim):
        output_size *= data.shape[d]

    input_layer = tf.placeholder(tf.float32, shape=(None, 1))

    W1 = tf.Variable(tf.random_normal([1, layer1_size], stddev=0.01))
    b1 = tf.Variable(tf.random_normal([layer1_size], stddev=0.01))
    layer1 = tf.matmul(input_layer, W1) + b1
    layer1 = tf.tanh(layer1)

    W2 = tf.Variable(tf.random_normal([layer1_size, output_size], stddev=0.01))
    b2 = tf.Variable(tf.random_normal([output_size], stddev=0.01))
    layer2 = tf.matmul(layer1, W2) + b2

    return input_layer, layer2

def create_1_256_network(data):
    """NETWORK ARCHITECTURE:
       Input        Output-Layer
       1 (lin)  ->  X*Y (lin)
       X, Y are the data dimensions
    """
    input_layer = tf.placeholder(tf.float32, shape=(None, 1))
    output_size = 1
    for d in range(1, data.ndim):
        output_size *= data.shape[d]

    W = tf.Variable(tf.random_normal([1, output_size], stddev=0.01))
    b = tf.Variable(tf.random_normal([output_size], stddev=0.01))

    return input_layer, input_layer * W + b

def create_1_32_deconv_256_network(data, **kwargs):
    """NETWORK ARCHITECTURE
       Input        Layer1          Output-Layer
       1 (lin)  ->  64 (tanh)   ->  X*Y (tanh(deconvolution))
       X, Y are the data dimensions
    """

    batch_size = kwargs["batch_size"]
    nr_output_channels = 1

    input_layer = tf.placeholder(tf.float32, shape=(None, 1))

    fcl_output_size = 64 #hardcoded for now
    W = tf.Variable(tf.random_normal([1, fcl_output_size], stddev=0.01))
    b = tf.Variable(tf.random_normal([fcl_output_size], stddev=0.01))

    # fully connected layer, reshaped to 2D
    nr_fcl_output_channels = 2 * nr_output_channels
    fc_layer = tf.tanh(W * input_layer + b)
    fc_layer = tf.reshape(fc_layer,
            [batch_size, 8, 4, nr_fcl_output_channels])

    # deconvolutional layer
    output_shape = [batch_size, 16, 8, nr_output_channels]
    strides = [1, 2, 2, 1]
    filter_weights = tf.Variable(tf.random_normal(
        [5, 5, nr_output_channels, nr_fcl_output_channels], stddev=0.01))
    deconv_layer = tf.nn.conv2d_transpose(
            fc_layer,
            filter_weights,
            output_shape,
            strides)

    return input_layer, tf.tanh(tf.reshape(deconv_layer,
            [batch_size, output_shape[1]*output_shape[2]]))
