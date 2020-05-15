import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets(r"D:\PycharmProjects\tensorflow_learning\MNIST_data", one_hot=True)

batch_size = 100

n_batch = mnist.train.num_examples // batch_size

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([1, 10]))
prediction = tf.nn.softmax(tf.matmul(x, W) + b)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
optimizer = tf.train.GradientDescentOptimizer(0.2)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(20):
        for batch in range(n_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={x: batch_x, y: batch_y})
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter"+str(epoch)+",Testing Accuracy "+str(acc))

