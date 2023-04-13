#!/usr/bin/env python

import tensorflow as tf
from tensorflow.keras import layers


class MyModel(tf.keras.Model):
    def __init__(self, hidden_dimension, hidden_dimension2, output_dimension):
        super(MyModel, self).__init__(name='my_model')
        self.dense_layer1 = layers.Dense(hidden_dimension, activation='relu')
        self.dense_layer2 = layers.Dense(hidden_dimension2, activation='relu')
        self.dense_layer3 = layers.Dense(output_dimension, activation='softmax')

    def call(self, inputs):
        x = self.dense_layer1(inputs)
        x = self.dense_layer2(x)

        return self.dense_layer3(x)


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

model = MyModel(64, 64, 10)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#model.fit(x_train, y_train, batch_size=64, epochs=3)
#model.evaluate(x_test, y_test, verbose=2)

