import tensorflow as tf

m1 = tf.constant([[3, 3]])
m2 = tf.constant([[2], [3]])
product = tf.matmul(m1, m2)
print(product)

with tf.Session() as sess:
    result = sess.run(product)
    print(result)
