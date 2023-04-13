#!/usr/bin/env python

# https://www.katacoda.com/basiafusinska/courses/tensorflow-getting-started/tensorflow-mnist-beginner

import os
import tensorflow.compat.v1 as tf
#from tensorflow.examples.tutorials.mnist import input_data
import tensorflow_datasets

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()


#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
mnist = tensorflow_datasets.load("mnist")

image_size = 28
labels_size = 10
learning_rate = 0.05
steps_number = 1000
batch_size = 100

# define placeholders
training_data = tf.placeholder(tf.float32, [None, image_size * image_size])
labels = tf.placeholder(tf.float32, [None, labels_size])

# variables to be tuned
W = tf.Variable(tf.truncated_normal([image_size * image_size, labels_size], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[labels_size]))
print("W=", W)
print("b=", b)

# build the network (only output layer)
output = tf.matmul(training_data, W) + b
print("output=", output)

# training step
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=output))
print("loss=", loss)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
print("train_step=", train_step)

# accuracy calculation
correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("correct_prediction=", correct_prediction)
print("accuracy=", accuracy)

# run the training
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(steps_number):
    # get the next batch
    input_batch, labels_batch = mnist.train.next_batch(batch_size)
    feed_dict = {training_data: input_batch, labels: labels_batch}
    
    # run the training step
    train_step.run(feed_dict=feed_dict)

    # print the accuracy progress on the batch every 100 steps
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict=feed_dict)
        print("step %d, training batch accuracy %g %%" % (i, train_accuracy * 100))

# evaluate on the test set
test_accuracy = accuracy.eval(feed_dict={training_data: mnist.test.images, labels: mnist.test.labels})
print("Test accuracy: %g %%" % (test_accuracy * 100))

