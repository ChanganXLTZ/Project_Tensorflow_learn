# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 19:34:59 2018

@author: LZC
"""

from __future__ import print_function

import tensorflow as tf

g1 = tf.Graph()
with g1.as_default():
    a = tf.constant([5,3,2,7,1,4])
    b = tf.constant([4,6,4])
    A1 = tf.reshape(a,[6,1])
    B1 = tf.reshape(b,[1,3])
    A2 = tf.reshape(a,[2,3])
    B2 = tf.reshape(b,[3,1])
    C1 = A1 * B1
    C2 = tf.matmul(A1,B1) # C1 C2 区别？？
    D = tf.matmul(A2,B2)
    with tf.Session() as Sess:
        print('\n\b=====计算图-1=====')
        print('张量a的初试状态：',a.eval())
        print('张量b的初试状态：',b.eval())
        print('张量a转换成：\n',A1.eval())
        print('张量b转换成：\n',B1.eval())
        print('张量乘积1:\n',C1.eval())
        print('张量乘积1:\n',C2.eval())
        print('张量乘积2：\n',D.eval())
g2 = tf.Graph()
with g2.as_default():
    # Create a variable with the initial value.
    v = tf.Variable([3,2,3,4,2,1,2])
    # Create a variable of shape [??], with a random initial value,
    # sampled from a normal distribution with mean 1 and standard deviation 0.35.
    w = tf.Variable(tf.random_normal([3,3], mean=1.0, stddev=0.35))
    # TensorFlow 的一个特性是变量初始化不是自动进行的。
    # 要初始化变量，最简单的方式是调用 global_variables_initializer。
    # 请注意 Session.run() 的用法（与 eval() 的用法大致相同）。
    initialization = tf.global_variables_initializer()
    with tf.Session() as Sess:
        Sess.run(initialization) # 执行初始化
        print('\n\b=====计算图-2-1=====')
        print('变量v：',v.eval())
        print('变量w：\n',w.eval())
g3 = tf.Graph()
with g3.as_default():
    u = tf.Variable(list(range(1,10)))
    with tf.Session() as Sess3:
        initialization3 = tf.global_variables_initializer()
        Sess3.run(initialization3)
        print('\n\b=====计算图-3-1=====')
        print('变量u：',u.eval())
print('\n注意图2和图3的初始化节点创建位置，图2再会话启动前，图3在会话中')
print('在第二次调用图时，尝试初始化')
# 初始化后，变量的值保留在同一会话中（不过，当您启动新会话时，需要重新初始化它们）
with g2.as_default():
    with tf.Session() as Sess2_2:
        print('\n\b=====计算图-2-2=====')
        Sess2_2.run(initialization)
        print('变量v：',v.eval())
with g3.as_default():
    with tf.Session() as Sess3_2:
        print('\n\b=====计算图-3-2=====')
        Sess3_2.run(initialization3)
        print('变量u：',u.eval())