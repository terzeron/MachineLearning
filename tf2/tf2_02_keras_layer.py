#!/usr/bin/env python

import tensorflow as tf

INPUT_SIZE = (20, 1)
inputs = tf.keras.layers.Input(shape=INPUT_SIZE)
print(inputs)
dropout = tf.keras.layers.Dropout(rate=0.2)(inputs)
print(dropout)
hidden = tf.keras.layers.Dense(units=10, activation=tf.nn.sigmoid)(dropout)
print(hidden)
output = tf.keras.layers.Dense(units=2, activation=tf.nn.sigmoid)(hidden)
print(output)

