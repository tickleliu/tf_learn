# -*- coding:utf-8 -*-
""" 
@author:mlliu
@file: tf_split.py 
@time: 2019/02/14 
"""

import numpy as np
import tensorflow as tf

a = np.reshape(np.arange(0, 64), (8, 8))
b = np.reshape(np.arange(0, 16), (4, 4))
c = np.concatenate((a,b))
print(a)
tf_a = tf.Variable(initial_value=a, dtype=tf.float32)
tf_as = tf.split(tf_a, num_or_size_splits=2, axis=0)
tf_as = tf.stack(tf_as, axis=0)
tf_as = tf.split(tf_as, num_or_size_splits=2, axis=1)
tf_as = tf.stack(tf_as, axis=1)
tf_bs = tf.keras.layers.GlobalMaxPool2D(data_format='channels_first')(tf_as)

sess = tf.Session()
with sess.as_default():
    init = tf.initialize_all_variables()
    sess.run(init)
    # print(sess.run(tf_a))
    k = sess.run(tf_as)
    print(k)
    print(k.shape)
    k = sess.run(tf_bs)
    print(k)
