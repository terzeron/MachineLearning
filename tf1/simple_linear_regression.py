#!/usr/bin/env python

import os
import tensorflow.compat.v1 as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

# x: [1, 2, 3]
# y: [1, 2, 3]
# y = 1.0 * x + 0.0인 비례관계의 기울기(1.0)과 y절편(0.0)을 선형회귀 모델로 구해냄

x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

# 텐서를 이용하여 1차원 회귀 모델을 구성함
hypothesis = W * X + b

# 비용함수는 가설과 계산된 y값의 차이의 제곱의 평균으로 정의함
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# 경사하강법으로 0.01의 비율로 최적화하여 비용이 가장 작은 Variable을 찾아내도록 정의함
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1000):
        _, cost_val = sess.run([train_op, cost], feed_dict={X: x_data, Y: y_data})

    print(step, cost_val, sess.run(W), sess.run(b))

    print("---- Test ----")
    print("X: 5, Y:", sess.run(hypothesis, feed_dict={X: 5}))
    print("X: 2.5, Y:", sess.run(hypothesis, feed_dict={X: 2.5}))
    print("X: 3001.532, Y:", sess.run(hypothesis, feed_dict={X: 3001.532}))

# step을 1000번 대신 100번만 반복하면 W와 b의 값의 정확도가 크게 떨어짐
