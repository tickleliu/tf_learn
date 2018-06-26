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

import tensorflow as tf

tf.flags.DEFINE_string("value","","")
FLAGS = tf.app.flags.FLAGS

def main():
    t1 = [[1, 2, 3], [4, 5, 6]]
    t2 = [[7, 8, 9], [10, 11, 12]]
    pass


if __name__ == "__main__":
    tf.app.run()
