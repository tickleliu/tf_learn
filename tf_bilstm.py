# -*- coding:utf-8 -*-
""" 
@author:mlliu
@file: tf_lstm.py 
@time: 2018/06/27 
"""

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用 GPU 0，1

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float("batch_size", 1, "iteration batch size")
tf.app.flags.DEFINE_float("time_step", 4, "lstm time step")
tf.app.flags.DEFINE_float("embedding_size", 1, "embedding size")
tf.app.flags.DEFINE_integer("hidden_num", 1, "lstm hidden unit number")


def main(_):
    # allow gpu memory growth
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4, allow_growth=True)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    # config.gpu_options = gpu_options

    # data
    x = np.linspace(0, 1000, 10000 * FLAGS.time_step * FLAGS.embedding_size).reshape(
        (10000, FLAGS.time_step, FLAGS.embedding_size))
    x = np.sin(x)
    n = np.random.randn(10000, 1) / 1000
    y = x[:, -1]
    # y = y + n
    l = np.ones(shape=[10000, 1], dtype=np.int32) * 2

    # model

    # xp = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, FLAGS.time_step, FLAGS.embedding_size])
    # yp = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, 1])
    # lp = tf.placeholder(dtype=tf.int32, shape=[FLAGS.batch_size])

    xp = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.time_step, FLAGS.embedding_size])
    yp = tf.placeholder(dtype=tf.float32, shape=[None, 1])
    lp = tf.placeholder(dtype=tf.int32, shape=[None])

    for i in range(2):
        with tf.variable_scope(None, default_name="bidirectional-rnn"):
            if i == 0:
                input = xp
            # print(tf.shape(input))
            # input = tf.Print(input, [tf.shape(input)])
            fw_lstm_cell = rnn.BasicLSTMCell(num_units=FLAGS.hidden_num, forget_bias=1.0,
                                             activation=tf.nn.tanh, state_is_tuple=True)
            fw_lstm_cell = rnn.DropoutWrapper(cell=fw_lstm_cell, input_keep_prob=1.0, output_keep_prob=0.7)

            bw_lstm_cell = rnn.BasicLSTMCell(num_units=FLAGS.hidden_num, forget_bias=1.0,
                                             activation=tf.nn.tanh, state_is_tuple=True)
            bw_lstm_cell = rnn.DropoutWrapper(cell=bw_lstm_cell, input_keep_prob=1.0, output_keep_prob=0.7)
            fw_init_state = fw_lstm_cell.zero_state(FLAGS.batch_size, dtype=tf.float32)
            bw_init_state = bw_lstm_cell.zero_state(FLAGS.batch_size, dtype=tf.float32)

            # print(layers)
            # print([lstm_cell] * 4)

            # init_state = lstm_cell.zero_state(FLAGS.batch_size, dtype=tf.float32)
            # (outputs, state) = tf.nn.dynamic_rnn(lstm_cell, inputs=xp, initial_state=init_state, dtype=tf.float32,
            #                                      time_major=False,
            #                                      sequence_length=lp)
            (outputs, state) = tf.nn.bidirectional_dynamic_rnn(fw_lstm_cell, bw_lstm_cell, inputs=input,
                                                               initial_state_fw=fw_init_state,
                                                               initial_state_bw=bw_init_state,
                                                               dtype=tf.float32,
                                                               sequence_length=lp)
            input = tf.concat(outputs, 2)
    outputs = tf.Print(input, [outputs], summarize=1000)
    # outputs = tf.Print(outputs, [input], summarize=1000)
    # outputs = tf.Print(outputs, [state], summarize=1000)

    batch_size = tf.shape(outputs)[0]
    max_length = tf.shape(outputs)[1]
    out_size = int(outputs.get_shape()[2])
    index = tf.range(0, batch_size) * max_length
    index = index + (lp - 1)
    flat = tf.reshape(outputs, [-1, out_size])
    outputs = tf.gather(flat, index)

    w = tf.Variable(initial_value=tf.truncated_normal(shape=[FLAGS.hidden_num * 2, 1]), dtype=tf.float32)
    b = tf.Variable(initial_value=tf.truncated_normal(shape=[1, 1]), dtype=tf.float32)

    out = tf.matmul(outputs, w)
    out = out + b

    tv = tf.trainable_variables()
    regularization_cost = 0.001 * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(out - yp))) + regularization_cost

    tf.summary.scalar("loss", loss)
    merged_summary = tf.summary.merge_all()
    # train_op = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
    global_step = tf.Variable(0, trainable=False)
    add_global = global_step.assign_add(1)
    learning_rate = tf.train.exponential_decay(learning_rate=0.01, global_step=global_step, decay_rate=0.9,
                                               decay_steps=10)
    op = tf.train.AdamOptimizer(learning_rate=learning_rate)
    grads_and_vars = op.compute_gradients(loss)
    grads_and_vars = [(gv[0] / 2, gv[1]) for gv in grads_and_vars]
    with tf.control_dependencies([add_global]):
        train_op = op.apply_gradients(grads_and_vars)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        writer = tf.summary.FileWriter("./", sess.graph)
        for i in range(20000):
            ids = np.arange(x.shape[0])
            np.random.shuffle(ids)
            ids = ids[0:FLAGS.batch_size]
            _, loss_value, summary = sess.run([train_op, loss, merged_summary],
                                              feed_dict={xp: x[ids], yp: y[ids], lp: l[ids].flatten()})
            print("迭代次数 %s, 训练误差 %s" % (i, loss_value))
            writer.add_summary(summary, i)
            # def dynamic_rnn(cell, inputs, sequence_length=None, initial_state=None,
            #                 dtype=None, parallel_iterations=None, swap_memory=False,
            #                 time_major=False, scope=None):


if __name__ == "__main__":
    tf.app.run()
