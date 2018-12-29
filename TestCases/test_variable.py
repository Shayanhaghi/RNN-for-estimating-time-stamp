import tensorflow as tf

firstTensor = tf.Variable(tf.random_uniform([2, 2], -1.0, 1.0), trainable=True)
objective = tf.square(firstTensor)
objective_mean = tf.reduce_mean(objective)

# create optimizer object
optimizer = tf.train.AdamOptimizer(1e-4)

# (optimize node minimize the objective mean)
optimize = optimizer.minimize(objective_mean)

# make an Interactive session :
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for i in range(20):
    output = sess.run([objective_mean, firstTensor, objective])
    print(output)
