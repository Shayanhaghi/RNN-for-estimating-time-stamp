import tensorflow as tf

x = tf.constant([[10, 2, 7], [5, 6, 8]])
x = tf.expand_dims(x, -1)
print(x)
y = tf.tile(x, [2, 1, 1])
session = tf.Session()
z = session.run(y)
print(z)
