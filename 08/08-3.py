import tensorflow as tf

with tf.Session() as sess:
    with tf.gfile.FastGFile('../inception/classify_image_graph_def.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    writer = tf.summary.FileWriter('../inception/log', sess.graph)
    writer.close()