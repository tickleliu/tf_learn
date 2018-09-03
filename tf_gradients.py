# -*- coding:utf-8 -*-
""" 
@author:mlliu
@file: tf_gradients.py 
@time: 2018/08/21 
"""

import tensorflow as tf
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 使用 GPU 0，1
A = tf.Variable(np.asarray(np.reshape([1, 2, 3, 4] * 4, newshape=(4, 4))), dtype=tf.float32)
C = tf.Variable(np.asarray(np.reshape([4, 3, 2, 1], newshape=(4, 1))), dtype=tf.float32)
B = tf.matmul(A, C)
B = tf.Print(B, [tf.shape(B)], message="B shape")

batch_size = 4
a_s = tf.split(A, num_or_size_splits=batch_size, axis=0)
cs = tf.split(C, num_or_size_splits=batch_size, axis=0)
bs = tf.split(B, num_or_size_splits=batch_size, axis=0)
gs = tf.gradients(xs=[A, C], ys=[B[0, 0]])
# gs = []
# for index in range(len(a_s)):
#     a = a_s[index]
#     b_s = bs[index]
#     c = cs[index]
#     b_s = tf.split(b_s, num_or_size_splits=batch_size, axis=1)
#     for b in b_s:
#         g = tf.gradients(xs=[a, c], ys=[b])
#         gs.append(g)
# gs = tf.concat(gs, axis=0)
# gs = tf.reshape(gs, shape=(4, 4, 4, 4))
sess = tf.Session()

with sess.as_default():
    init = tf.global_variables_initializer()
sess.run(init)
B_, gs_ = sess.run([B, gs])
print(B_, gs_)
print(tf.shape(B))
