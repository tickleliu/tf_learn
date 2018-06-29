# -*- coding:utf-8 -*-
""" 
@author:mlliu
@file: tf_loss.py 
@time: 2018/06/28 
"""

import tensorflow as tf
from tensorflow.contrib import legacy_seq2seq


def main(_):
    A = tf.random_normal([5, 4], dtype=tf.float32)
    A = tf.Print(A, [A], summarize=1999)
    B = tf.constant([1, 2, 1, 3, 3], dtype=tf.int32)
    C = tf.ones([5], dtype=tf.float32)
    # D = legacy_seq2seq.sequence_loss_by_example(tf.expand_dims(A, 0), tf.expand_dims(B, 1), tf.expand_dims(C, 1))
    D = legacy_seq2seq.sequence_loss_by_example([A], [B], [C])

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        r = sess.run(D)
        print(r)
        print(tf.local_variables())
        print(tf.global_variables())


if __name__ == "__main__":
    tf.app.run()
