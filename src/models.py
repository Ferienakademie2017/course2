import tensorflow as tf

def simple_model_1(x):
    #fc1_w = tf.get_variable("fc1_w", initializer=tf.random_normal([1, 256], stddev=0.1))
    #fc1_b = tf.get_variable("fc1_b", initializer=tf.constant(1.0, shape=[256]))
    #fc1_z = tf.add(tf.matmul(x, fc1_w), fc1_b)
    #fc1_a = tf.tanh(fc1_z)
    # fc1_a = tf.dropout(fc1_a, 0.5)
    fc1_a = tf.layers.dense(x, 256)  # activation = tf.tanh
    y_pred = tf.reshape(fc1_a, [-1, 16, 8, 2])
    return y_pred

def simple_loss_1(y_pred, y, flagField):
    loss = tf.reduce_mean(tf.abs(y_pred - y))
    return loss

