#!/usr/bin/env python  
# encoding: utf-8  

""" 
@version: v1.0 
@author: tickleliu  
@contact: tickleliu@163.com
@site: https://github.com/tickleliu 
@software: PyCharm 
@file: tf_flags.py 
@time: 2018/6/26 22:23 
"""
import tensorflow as tf

tf.app.flags.DEFINE_float("flag_float", 0.01, "input a float")
tf.app.flags.DEFINE_integer("flag_integer", 1, "input a int")
tf.app.flags.DEFINE_string("flag_string", "test", "input a string")

FLAGS = tf.app.flags.FLAGS

def main(_):
    # FLAGS.flag_values_dict()
    print(FLAGS.flag_string)




if __name__ == "__main__":
    # main()
    tf.app.run()
