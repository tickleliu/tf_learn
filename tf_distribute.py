# -*- coding:utf-8 -*-
""" 
@author:mlliu
@file: tf_distribute.py 
@time: 2018/09/07 
"""

import tensorflow as tf
import os

cluster = tf.train.ClusterSpec({
    "worker": ["10.0.0.244:2223"],
    "ps": ["10.0.0.244:2221"]
})
tf.app.flags.DEFINE_boolean("isps", 0, "input a string")
tf.app.flags.DEFINE_string("gpu", "1", "balance the training data")
FLAGS = tf.app.flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

isps = FLAGS.isps

if isps:
    server = tf.train.Server(cluster, job_name="ps", task_index=0)
    server.join()

else:
    server = tf.train.Server(cluster, job_name="worker", task_index=0)
    with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:0", cluster=cluster)):
        w = tf.get_variable('w', (2,2))
        b = tf.get_variable('b', (2, 2), tf.float32, initializer=tf.constant_initializer(5))
        addwb = w + b
        mutwb = w * b
        divwb = w / b

saver = tf.train.Saver()
init_op = tf.initialize_all_variables()
sv = tf.train.Supervisor(init_op=init_op, saver=saver)

with sv.managed_session(server.target) as sess:
    while 1:
        print(sess.run([addwb, mutwb, divwb]))
