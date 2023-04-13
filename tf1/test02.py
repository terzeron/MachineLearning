#!/usr/bin/env python

import os
import tensorflow.compat.v1 as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#tf.disable_v2_behavior()


with tf.Session() as sess:
    x = tf.Variable([1.0, 2.0])
    a = tf.constant([3.0, 3.0])
    x.initializer.run()

    sub = tf.subtract(x, a)
    print(sub.eval())

