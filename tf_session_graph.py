# -*- coding:utf-8 -*-
""" 
@author:mlliu
@file: tf_session_graph.py 
@time: 2018/12/20 
"""

import tensorflow as tf

with tf.variable_scope("level1"):
    a0 = tf.Variable(0, name="a")
init0 = tf.initialize_all_variables()
print(a0.name)
print(a0.graph)

g1 = tf.Graph()
print(g1)
g2 = tf.Graph()
print(g2)
s0 = tf.Session(graph=tf.get_default_graph())
s1 = tf.Session(graph=g1)
s21 = tf.Session(graph=g2)
s22 = tf.Session(graph=g2)

with g1.as_default():
    with s1.as_default():
        with tf.variable_scope("level1"):
            a1 = tf.Variable(2, name="a")
            print(a1.name)
            a1_add = a1.assign(10)
        init1 = tf.initialize_all_variables()
with g2.as_default():
    with s21.as_default():
        with tf.variable_scope("level1"):
            a21 = tf.Variable(21, name="a", dtype=tf.int32)
            a22 = tf.get_variable(name="a", initializer=2, dtype=tf.int32)

    with s22.as_default():
        with tf.variable_scope("level1", reuse=tf.AUTO_REUSE):
            a23 = tf.get_variable("a", initializer=1, dtype=tf.int32)
    print(a21.name)
    print(a22.name)
    print(a23.name)
    init2 = tf.initialize_all_variables()

s0.run(init0)  # 初始化函数，是属于某个特定的graph的
s1.run(init1)
s21.run(init2)
s22.run(init2)

print(a0.eval(s0))
print(a1.eval(s1))
print(a21.eval(s22))
print(a22.eval(s22))
print(a23.eval(s21))
s21.close()
print(a21.eval(s22))
