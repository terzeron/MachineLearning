#!/usr/bin/env python

import numpy as np
import tensorflow as tf
from tensorflow.keras import preprocessing, layers
from tensorflow.keras.preprocessing.sequence import pad_sequences


samples = ['너 오늘 이뻐 보인다', '나는 오늘 기분이 더러워', '끝내주는데, 좋은 일이 있나봐', '나 좋은 일이 생겼어', '아 오늘 진짜 짜증나', '환상적인데, 정말 좋은거 같아']
labels = [[1], [0], [1], [1], [0], [1]]
tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(samples)
sequences = tokenizer.texts_to_sequences(samples)
MAXLEN = 4
X = pad_sequences(sequences, maxlen=MAXLEN, padding='post', truncating='post')
y = np.array(labels, dtype=np.float32)

batch_size = 2
num_epochs= 10
vocab_size = len(tokenizer.word_index) + 1
emb_size = 128
hidden_dimension = 256
output_dimension = 1

model = tf.keras.Sequential()
model.add(layers.Embedding(vocab_size, emb_size, input_length=4))
model.add(layers.Lambda(lambda x: tf.reduce_mean(x, axis=1)))
model.add(layers.Dense(hidden_dimension, activation='relu'))
model.add(layers.Dense(output_dimension, activation='sigmoid'))
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X, y, epochs=num_epochs, batch_size=batch_size)

class CustomModel(tf.keras.Model):
    def __init__(self, vocab_size, embed_dimension, hidden_dimension, output_dimension):
        super(CustomModel, self).__init__(name="my_model")
        self.embedding = layers.Embedding(vocab_size, embed_dimension)
        self.dense_layer = layers.Dense(hidden_dimension, activation='relu')
        self.output_layer = layers.Dense(output_dimension, activation='sigmoid')

    def call(self, inputs):
        x = self.embedding(inputs)
        x = tf.reduce_mean(x, axis=1)
        x = self.dense_layer(x)
        x = self.output_layer(x)
        return x

model = CustomModel(vocab_size=vocab_size, embed_dimension=emb_size, hidden_dimension=hidden_dimension, output_dimension=output_dimension)
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X, y, epochs=num_epochs, batch_size=batch_size)
