#!/usr/bin/env python

import os
import tensorflow.compat.v1 as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#tf.disable_v2_behavior()


# rank
# 3: rank 0, shape []
# [1, 2, 3]: rank 1, shape [3]
# [[1, 2, 3], [4, 5, 6]]: rank 2, shape [2, 3]
# [[[1, 2, 3]], [[7, 8, 9]]]: rank 3, shape [2, 1, 3]

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
print(node1, node2)

with tf.Session() as sess:
    print(sess.run([node1, node2]))

    node3 = tf.add(node1, node2)
    print("node3:", node3)

    print(sess.run(node3))

    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    adder_node = a + b
    print(sess.run(adder_node, {a: 3, b: 4.5}))
    print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))
