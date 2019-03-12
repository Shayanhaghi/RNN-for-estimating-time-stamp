from feeding_network import BatchFeeder
import numpy as np
import tensorflow as tf


# batchFeeder = BatchFeeder()
# batchFeeder.set_user_max_train_max_value()
# print("********-----------------------********")
# value = batchFeeder.create_batch()

# print(np.mean(value[0]))
# print(np.max(value[0]))
# print(np.min(value[0]))


# data_tf = tf.convert_to_tensor(value[0], np.float32)
#
# tf.summary.histogram(data_tf)
#
# print(np.max(np.log(value[0])))

# sess = tf.InteractiveSession()
# print(data_tf.eval())
# print(batchFeeder)
# print(np.log(100))
def test_func(a):
    return a + 2


x = list()
x.append(10)
x.append(2)
print(x)
y = np.asarray(x)
print(y)
print(max(3, 4))
for i in range(100):
    print(i)
import sys

print(dir(sys.modules[__name__]))
print(eval("test_func")(3))
x = [3, 4, 6]
print(x[::2])
for i in range(1, 10):
    print(i)


def test_time(new_time_stamp, old_time_stamp):
    pass