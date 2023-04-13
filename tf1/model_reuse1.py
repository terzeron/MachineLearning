#!/usr/bin/env python

import tensorflow.compat.v1 as tf
import numpy as np
#tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

data = np.loadtxt('data.csv', delimiter=',', unpack=True, dtype='float32')
x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])

# 학습 횟수를 저장하는 변수
global_step = tf.Variable(0, trainable=False, name='global_step')

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([2, 10], -1.0, 1.0))
L1 = tf.nn.relu(tf.matmul(X, W1))

W2 = tf.Variable(tf.random_uniform([10, 20], -1.0, 1.0))
L2 = tf.nn.relu(tf.matmul(L1, W2))

W3 = tf.Variable(tf.random_uniform([20, 3], -1.0, 1.0))
model = tf.nn.relu(tf.matmul(L2, W3))

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model))

optimizer = tf.train.AdamOptimizer(learning_rate=0.05)
train_op = optimizer.minimize(cost, global_step=global_step)

sess = tf.Session()
# 앞서 정의했던 변수들을 가져옴
saver = tf.train.Saver(tf.global_variables())

ckpt = tf.train.get_checkpoint_state('model')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())

for step in range(100):
    sess.run(train_op, feed_dict={X:x_data, Y:y_data})
    print("step: %d, " % sess.run(global_step), "cost: %.3f" % sess.run(cost, feed_dict={X:x_data, Y:y_data}))

saver.save(sess, 'model/dnn.ckpt', global_step=global_step)

prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
print("prediction: ", sess.run(prediction, feed_dict={X:x_data}))
print("target: ", sess.run(target, feed_dict={Y:y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print("accuracy: %.2f" % sess.run(accuracy * 100, feed_dict={X:x_data, Y:y_data}))

    
