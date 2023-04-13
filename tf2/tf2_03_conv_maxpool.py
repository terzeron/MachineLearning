#!/usr/bin/env python

import tensorflow as tf

INPUT_SIZE = (28, 28)
inputs = tf.keras.Input(shape=INPUT_SIZE)
dropout = tf.keras.layers.Dropout(rate=0.2)(inputs)
conv = tf.keras.layers.Conv1D(filters=10, kernel_size=3, padding='same', activation=tf.nn.relu)(dropout)
max_pool = tf.keras.layers.MaxPool1D(pool_size=3, padding='same')(conv)
flatten = tf.keras.layers.Flatten()(max_pool)
hidden = tf.keras.layers.Dense(units=50, activation=tf.nn.relu)(flatten)
output = tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)(hidden)
