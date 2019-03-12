import tensorflow as tf
import numpy as np
print(np.log10(10))
print(np.log(10))
# tf.enable_eager_execution()
x = tf.log(tf.constant(10.0))

session = tf.Session()
print(session.run(x))