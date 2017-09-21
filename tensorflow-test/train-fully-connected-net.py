#importing libraries
import tensorflow as tf
import numpy as np

#setting up a session
sess = tf.InteractiveSession()

#setting up the input to the network
input_size = 1
in_node = tf.placeholder(tf.float32)

#setting up the network architecture with all the weights (=variables)
layer_size = 3
W1 = tf.Variable(tf.zeros([input_size,layer_size]))
W2 = tf.Variable(tf.zeros([layer_size,layer_size]))
b1 = tf.Variable(tf.zeros([layer_size]))
b2 = tf.Variable(tf.zeros([layer_size]))

layer1 = in_node*W1 + b1
layer2 = tf.matmul(layer1,W2) + b2
output = layer2

#initializing the variables
init = tf.global_variables_initializer()
sess.run(init)

#implementing the loss function
desired_output = tf.placeholder(tf.float32)
loss = tf.reduce_sum(tf.square(output - desired_output))



#y = tf.placeholder(tf.float32)
#squared_deltas = tf.square(linear_model - y)
#loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {in_node: [0, 0.5, 1], desired_output: [[1,0,0], [0,1,0], [0,0,1] ]}))
print(sess.run(output, {in_node: [0, 0.5, 1], desired_output: [[1,0,0], [0,1,0], [0,0,1] ]}))


#training the network
#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
