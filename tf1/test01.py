#!/usr/bin/env python

import os
import tensorflow.compat.v1 as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#tf.disable_v2_behavior()

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.], [2.]])

product = tf.matmul(matrix1, matrix2)
print(product)

#with tf.Session() as sess:
    #result = sess.run(product)
    #tf.print(result)
