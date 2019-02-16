# -*- coding:utf-8 -*-
""" 
@author:mlliu
@file: tf_dynamic_rnn.py 
@time: 2019/02/16 
"""

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用 GPU 0，1

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("batch_size", 2, "iteration batch size")
tf.app.flags.DEFINE_integer("time_step", 8, "lstm time step")
tf.app.flags.DEFINE_integer("embedding_size", 1, "embedding size")
tf.app.flags.DEFINE_integer("hidden_num", 1, "lstm hidden unit number")


def main(_):
    x = np.asarray([[1, 2, 3, 4, 0, 0, 0, 0], [1, 2, 3, 4, 0, 0, 0, 0]])
    x = np.reshape(x, newshape=(2, 8, 1))
    l = np.asarray([4, 4], dtype=np.int32)
    l = np.reshape(l, newshape=[2])

    xp = tf.placeholder(dtype=tf.float32, shape=[None, 8, 1])
    lp = tf.placeholder(dtype=tf.int32, shape=[None])

    for i in range(1):
        with tf.variable_scope(None, default_name="bidirectional-rnn"):
            if i == 0:
                input = xp
            fw_lstm_cell = rnn.BasicLSTMCell(num_units=FLAGS.hidden_num, forget_bias=1.0,
                                             activation=tf.nn.tanh, state_is_tuple=True)
            fw_lstm_cell = rnn.DropoutWrapper(cell=fw_lstm_cell, input_keep_prob=1.0, output_keep_prob=0.7)

            bw_lstm_cell = rnn.BasicLSTMCell(num_units=FLAGS.hidden_num, forget_bias=1.0,
                                             activation=tf.nn.tanh, state_is_tuple=True)
            bw_lstm_cell = rnn.DropoutWrapper(cell=bw_lstm_cell, input_keep_prob=1.0, output_keep_prob=0.7)
            fw_init_state = fw_lstm_cell.zero_state(FLAGS.batch_size, dtype=tf.float32)
            bw_init_state = bw_lstm_cell.zero_state(FLAGS.batch_size, dtype=tf.float32)

            (outputs, state) = tf.nn.bidirectional_dynamic_rnn(fw_lstm_cell, bw_lstm_cell, inputs=input,
                                                               initial_state_fw=fw_init_state,
                                                               initial_state_bw=bw_init_state,
                                                               dtype=tf.float32,
                                                               sequence_length=lp)
            input = tf.concat(outputs, 2)
    outputs = tf.Print(input, [outputs], summarize=1000)

    batch_size = tf.shape(outputs)[0]
    max_length = tf.shape(outputs)[1]
    out_size = int(outputs.get_shape()[2])
    index = tf.range(0, batch_size) * max_length
    index = index + (lp - 1)
    flat = tf.reshape(outputs, [-1, out_size])
    outputs_result = tf.gather(flat, index)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        o, o_r = sess.run([outputs, outputs_result],
                          feed_dict={xp: x, lp: l.flatten()})
        print(o)
        print(o_r)


if __name__ == "__main__":
    tf.app.run()
