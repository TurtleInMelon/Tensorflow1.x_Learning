import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 中间层
weights1 = tf.Variable(tf.random_normal([1, 10]))
b1 = tf.Variable(tf.zeros([1, 10]))
o1 = tf.matmul(x, weights1) + b1
o2 = tf.nn.tanh(o1)

# 输出层
weight2 = tf.Variable(tf.random_normal([10, 1]))
b2 = tf.Variable(tf.zeros([1, 1]))
o3 = tf.matmul(o2, weight2) + b2
prediction = tf.nn.tanh(o3)

loss = tf.reduce_mean(tf.square(y - prediction))
optimizer = tf.train.GradientDescentOptimizer(0.1)  # 梯度下降优化器
train = optimizer.minimize(loss)    # 最小化代价函数

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for _ in range(2000):
        sess.run(train, feed_dict={x: x_data, y: y_data})

    prediction_value = sess.run(prediction, feed_dict={x: x_data})
    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, prediction_value, 'g-', lw=5)
    plt.show()
