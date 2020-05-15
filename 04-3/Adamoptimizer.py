import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data



batch_size = 100

n_batch = mnist.train.num_examples // batch_size

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
learning_rate = tf.Variable(0.001, dtype=tf.float32)

W1 = tf.Variable(tf.truncated_normal([784, 200], stddev=0.1))
b1 = tf.Variable(tf.zeros([1, 200]))
o1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
o1_drop = tf.nn.dropout(o1, keep_prob)

W2 = tf.Variable(tf.truncated_normal([200, 100], stddev=0.1))
b2 = tf.Variable(tf.zeros([1, 100]))
o2 = tf.nn.tanh(tf.matmul(o1_drop, W2) + b2)
o2_drop = tf.nn.dropout(o2, keep_prob)

W3 = tf.Variable(tf.truncated_normal([100, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([1, 10]))
prediction = tf.nn.softmax(tf.matmul(o2_drop, W3) + b3)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(50):
        sess.run(tf.assign(learning_rate, 0.001 * (0.95 ** epoch)))
        for batch in range(n_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={x: batch_x, y: batch_y, keep_prob: 1})
        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 1.0})
        print("Iter" + str(epoch) + ",Testing Accuracy " + str(test_acc) + "Training Accuracy " + str(train_acc))

