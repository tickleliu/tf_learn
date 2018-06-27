# -*- coding:utf-8 -*-
""" 
@author:mlliu
@file: tf_saver.py 
@time: 2018/06/27 
"""

import os
import numpy as np
import tensorflow as tf

tf.app.flags.DEFINE_string("save_path", "./", "model save path")

FLAGS = tf.app.flags.FLAGS


def main(_):
    # data
    x = np.linspace(0, 10, num=1000)
    x = np.expand_dims(x, 1)
    n = np.random.randn(x.shape[0], 1)
    y = np.sqrt(x) * 5 + n
    # y = x * 5 + n

    # model
    xp = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="x")
    yp = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="y")

    w1 = tf.Variable(tf.truncated_normal(shape=[1, 10], mean=0.5, stddev=1))
    b1 = tf.Variable(tf.truncated_normal(shape=[1, 10], mean=0.1, stddev=1))
    m_w1_p_b1 = tf.matmul(xp, w1) + b1
    l1 = tf.nn.relu(m_w1_p_b1)

    w2 = tf.Variable(tf.truncated_normal(shape=[10, 1], mean=0.5, stddev=1))
    b2 = tf.Variable(tf.truncated_normal(shape=[1, 1], mean=0.1, stddev=1))
    m_w2_p_b2 = tf.matmul(l1, w2) + b2
    out = tf.nn.relu(m_w2_p_b2)

    # loss
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(out - yp), reduction_indices=[1]))
    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

    saver = tf.train.Saver()

    with tf.Session() as sess:

        if os.path.exists(os.path.join(FLAGS.save_path, "file.ckpt")):
            saver.resore(sess, os.path.join(FLAGS.save_path, "file.ckpt"))
        else:
            sess.run([tf.local_variables_initializer(), tf.global_variables_initializer()])

        for i in range(1000):
            ids = np.arange(len(x))
            np.random.shuffle(ids)
            ids = ids[0: 10]
            _, loss_value = sess.run([train_op, loss], feed_dict={xp: x[ids], yp: y[ids]})
            saver.save(sess, os.path.join(FLAGS.save_path, ""))
            print("迭代次数：%d , 训练损失：%s" % (i, loss_value))


if __name__ == "__main__":
    tf.app.run()
