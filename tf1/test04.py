#!/usr/bin/env python

import os
import tensorflow.compat.v1 as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#tf.disable_v2_behavior()


input1 = tf.constant([3.0])
input2 = tf.constant([2.0])
input3 = tf.constant([5.0])

intermed = tf.add(input2, input3)
mul = tf.multiply(input1, intermed)

with tf.Session() as sess:
    result = sess.run([mul, intermed])
    print(result)
                        
