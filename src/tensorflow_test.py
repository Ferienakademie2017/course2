import tensorflow as tf
import numpy as np
import models



x = tf.placeholder(tf.float32, shape=[None, 1])
y = tf.placeholder(tf.float32, shape=[None, 16, 8, 2])
y_pred = models.simple_model_1(x)
target = np.ones((2, 16, 8, 2), dtype=np.float32)
loss = tf.reduce_mean(tf.abs(y_pred - y))
opt = tf.train.AdamOptimizer(0.001).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
# loss_result = sess.run([loss], feed_dict={x: np.array([[1.0]]), y: target})
loss_result = -1
for i in range(10000):
    _, loss_result = sess.run([opt, loss], feed_dict={x: np.array([[1.0]]), y: target})
print(loss_result)


