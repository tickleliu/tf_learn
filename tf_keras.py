# -*- coding:utf-8 -*-
""" 
@author:mlliu
@file: tf_keras.py 
@time: 2018/11/30 
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

img = tf.placeholder(dtype=tf.float32, shape=(None, 784))
labels = tf.placeholder(dtype=tf.float32, shape=(None, 10))

x = tf.keras.layers.Dense(128, activation="relu")(img)
x = tf.keras.layers.Dense(128, activation="relu")(x)
pred = tf.keras.layers.Dense(10, activation="softmax")(x)

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=labels))
train_step = tf.train.AdamOptimizer().minimize(loss=loss)

mnist_data = input_data.read_data_sets("/home/liuml/model_data/mnist", one_hot=True)

with tf.Session() as session:
    init = tf.global_variables_initializer()
    session.run(init)
    for _ in range(1000):
        batch_x, batch_y = mnist_data.train.next_batch(50)
        _, loss_val = session.run([train_step, loss], feed_dict={img: batch_x, labels: batch_y})
        print(loss_val)
