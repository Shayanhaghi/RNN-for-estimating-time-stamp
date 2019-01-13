import tensorflow as tf
import numpy as np


def create_network():
    user_lstm_state_reference = tf.Variable(tf.random_uniform([4, 2], -1.0, 1.0),
                                            trainable=True)
    users = tf.placeholder(tf.int32, shape=(2, 2), name="users")
    out = tf.nn.embedding_lookup(user_lstm_state_reference, users)
    return users, out


users, output = create_network()
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
x1 = np.array([[0, 1], [2, 3]], int)
x2 = np.array([[0, 1], [1, 0]], int)
s = sess.run([output], feed_dict={users: x1})
print(s[0])

s = sess.run([output], feed_dict={users: x2})
print(s[0])
