#!/usr/bin/env python

import tensorflow as tf
from tensorflow.keras import layers

inputs = tf.keras.Input(shape=(32,))
x = layers.Dense(64, activation='relu')(inputs)
y = layers.Dense(64, activation='relu')(x)
predictions = layers.Dense(10, activation='softmax')(x)

