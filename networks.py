"""Factory methods to create networks (i.e. the input and the output layer).
Input: single scalar. Output: flat array, dimensions derived from 'data'
argument."""

import tensorflow as tf


def create_1_8_256_network(data):
    #NETWORK ARCHITECTURE:
    #   Input        Layer1          Output-Layer
    #   1 (lin)  ->  8 (tanh)   ->   256 (lin)
    input_size = 1
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
    input_layer = tf.placeholder(tf.float32, shape=(None, 1))
    output_size = 1
    for d in range(1, data.ndim):
        output_size *= data.shape[d]

    W = tf.Variable(tf.random_normal([1, output_size], stddev=0.01))
    b = tf.Variable(tf.random_normal([output_size], stddev=0.01))

    return input_layer, input_layer * W + b
