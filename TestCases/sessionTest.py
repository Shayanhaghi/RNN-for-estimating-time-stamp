import tensorflow as tf
import numpy as np

# Build a graph.
a = tf.placeholder(tf.float64, shape=[1])
b = tf.placeholder(tf.float64, shape=[1])
inp = tf.concat([a, b], axis=0)
# weight = tf.trainable_variables(np.zeros([2, 2]))
# v1 = tf.get_variable("mv", initializer= init)
# v2 = tf.get_variable("mv2", initializer= init2)
# weight2 = tf.trainable_variables(np.zeros([2, 2]))
v1 = tf.Variable(np.array([2, 2]))
v2 = tf.Variable(np.array([2, 2]))
x = tf.matmul(v1, inp)
x1 = tf.sigmoid(x)
y = tf.multiply(v2, x1)
z = tf.train.AdamOptimizer(1e-4).minimize(y, name="optimizer2Optimize")
# Launch the graph in a session.
tf.global_variables_initializer()
sess = tf.InteractiveSession()

# Evaluate the tensor `c`.
sess.run(z, feed_dict={a: [3], b: [4]})
