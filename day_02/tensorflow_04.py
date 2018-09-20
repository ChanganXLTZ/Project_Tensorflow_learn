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
    initialization2 = tf.global_variables_initializer()
    with tf.Session() as Sess2_1:
        Sess2_1.run(initialization2) # 执行初始化
        print('\n\b=====计算图-2-1=====')
        print('变量v：',v.eval())
        print('变量w：\n',w.eval())
g3 = tf.Graph()
with g3.as_default():
    u = tf.Variable(list(range(1,10)))
    with tf.Session() as Sess3_1:
        initialization3 = tf.global_variables_initializer()
        Sess3_1.run(initialization3)
        print('\n\b=====计算图-3-1=====')
        print('变量u：',u.eval())
print('\n注意图2和图3的初始化节点创建位置，图2再会话启动前，图3在会话中')
print('在第二次调用图时，尝试初始化')
# 初始化后，变量的值保留在同一会话中（不过，当您启动新会话时，需要重新初始化它们）
with g2.as_default():
    with tf.Session() as Sess2_2:
        print('\n\b=====计算图-2-2=====')
        Sess2_2.run(initialization2)
        print('变量v：',v.eval())
        print('变量w：\n',w.eval())
        print('事实证明初始化再次执行了随机赋值函数')
with g3.as_default():
    with tf.Session() as Sess3_2:
        print('\n\b=====计算图-3-2=====')
        Sess3_2.run(initialization3)
        print('变量u：',u.eval())
print('\n没区别……')

# 要更改变量的值，请使用 assign 指令。
# 请注意，仅创建 assign 指令不会起到任何作用。
# 和初始化一样，您必须运行赋值指令才能更新变量值

g4 = tf.Graph()
with g4.as_default():
    data_4 = tf.Variable([1,2,3,4,5])
    with tf.Session() as sess:
        print('\n\b=====计算图-4=====')
        sess.run(tf.global_variables_initializer())
        print('初始化后的值：',data_4.eval())
        assignment = tf.assign(data_4[3],100)
        print('定义了修改值的操作未执行：',data_4.eval())
        sess.run(assignment)
        print('执行完修改值操作：',data_4.eval())

g5 = tf.Graph()
with g5.as_default():
    data_5 = tf.Variable(1,name = 'data5')
    init = tf.global_variables_initializer()
    assignment = tf.assign(data_5,10)
    with tf.Session() as sess:
        print('\n\b=====计算图-5=====')
        sess.run(init) # 初始化
        print('会话仅执行不定义')
        print('初始化后的值：',data_5.eval())
        sess.run(assignment) # 修值
        print('执行完修改值操作：',data_5.eval())
        
# 教程中的例子
with tf.Graph().as_default(), tf.Session() as sess:

    '''
Task 2: Simulate 10 throws of two dice. Store the results
 in a 10x3 matrix.​
We're going to place dice throws inside two separate
 10x1 matrices. We could have placed dice throws inside
 a single 10x2 matrix, but adding different columns of
 the same matrix is tricky. We also could have placed
 dice throws inside two 1-D tensors (vectors); doing so
 would require transposing the result.
 
创建一个骰子模拟，在模拟中生成一个 10x3 二维张量，其中：
列 1 和 2 均存储一个骰子的一次投掷值。
列 3 存储同一行中列 1 和 2 的值的总和。
例如，第一行中可能会包含以下值：
列 1 存储 4
列 2 存储 3
列 3 存储 7
    '''
    dice1 = tf.Variable(tf.random_uniform([10, 1],
                                        minval=1, maxval=7,
                                        dtype=tf.int32))
    dice2 = tf.Variable(tf.random_uniform([10, 1],
                                        minval=1, maxval=7,
                                        dtype=tf.int32))

    # We may add dice1 and dice2 since they share the same shape
    # and size.
    dice_sum = tf.add(dice1, dice2)

    # We've got three separate 10x1 matrices. To produce a single
    # 10x3 matrix, we'll concatenate them along dimension 1.
    resulting_matrix = tf.concat(
            values=[dice1, dice2, dice_sum], axis=1)

    # The variables haven't been initialized within the graph yet,
    # so let's remedy that.
    sess.run(tf.global_variables_initializer())
    print('\n\b=====计算图-6=====')
    print('d1 d2 sum')
    print(resulting_matrix.eval())
  