#-*- coding:utf-8 -*-
""" 
@author:mlliu
@file: tf_lstm_crf.py 
@time: 2018/06/29 
"""

import tensorflow as tf

score = tf.constant([2, 2])
tf.contrib.crf.viterbi_decode()