#!/usr/bin/env python  
# encoding: utf-8  

""" 
@version: v1.0 
@author: tickleliu  
@contact: tickleliu@163.com
@site: https://github.com/tickleliu 
@software: PyCharm 
@file: tf_trans.py 
@time: 2018/6/26 22:58 
"""

import numpy as np
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

tf.flags.DEFINE_string("value", "", "")
FLAGS = tf.app.flags.FLAGS


def main(_):
    sess = tf.InteractiveSession()
    t1 = [[1, 2, 3], [4, 5, 6]]
    t2 = [[7, 8, 9], [10, 11, 12]]
    # result = tf.reduce_sum(t1, axis=1)
    # print(result.eval())
    #
    # con0 = tf.concat([t1, t2], axis=0)
    # con1 = tf.concat([t1, t2], axis=1)
    # print(con0.eval())
    # print(con1.eval())
    #
    # t3 = [t1]
    # sque = tf.squeeze(t3, 0)
    # print(sque.eval())
    #
    # splits = tf.split(t1, 3, 1)
    # print([split.eval() for split in splits])
    #
    # mat = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape([3, -1])
    # print(mat)
    # ids = [[1, 2], [0, 1]]
    # res = tf.nn.embedding_lookup(mat, ids)
    # print(res.eval())
    #
    # print(tf.shape(t1))
    # t3 = tf.expand_dims(t1, 1)
    # print(t3.eval())
    # print(tf.squeeze(t3, squeeze_dims=[1]).eval())

    t1 = tf.expand_dims(t1, 2)
    print(t1.eval())
    print("slice %s" % tf.slice(t1, [0, 2, 0], [2, 1, 1]).eval())

    # print(tf.stack([t1, t2], axis=2).eval())
    #
    # pad = tf.pad(t1, [[1, 1], [1, 1]])
    # print(pad.eval())
    #
    # temp1 = tf.range(0, 10) + tf.constant(1, shape=[10])
    # temp2 = tf.gather(temp1, [[1, 2, 3], [7, 8, 9]])
    # print(temp2.eval())
    var = tf.Variable(initial_value=np.random.randn(4, 4), trainable=True, dtype=tf.float32)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    print(var.eval())


if __name__ == "__main__":
    tf.app.run()
