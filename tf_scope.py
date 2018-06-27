#-*- coding:utf-8 -*-  
""" 
@author:mlliu
@file: tf_scope.py 
@time: 2018/06/27 
"""

import tensorflow as tf


def main(_):
    with tf.name_scope("name") as name_scope:
        var1 = tf.get_variable(name="var1", shape=[2, 10])
        tf.get_variable_scope().reuse_variables()
        var2 = tf.get_variable(name="var1", shape=[2, 10])
        print(name_scope)
        print(var1.name)
        print(var2.name)

if __name__ == "__main__":
    tf.app.run()
