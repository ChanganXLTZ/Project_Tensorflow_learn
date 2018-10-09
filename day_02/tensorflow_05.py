# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 09:06:25 2018

@author: LZC
"""
from __future__ import print_function
import tensorflow as tf 
import numpy as np

d_size = 500
data_x = np.float32(np.random.rand(2,d_size)) # 生成随机数,2x100
data_y = np.dot([0.100,0.200],data_x)+0.300 # dot()返回的是两个数组的点积

# 构造一个线性模型
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0)) #  1行2列，范围(-1,1)的参数矩阵
y = tf.matmul(W, data_x) + b # 线性模型
y_test = data_x + b

# 最小化方差
loss = tf.reduce_mean(tf.square(y - data_y))
optimizer = tf.train.GradientDescentOptimizer(0.4)
train = optimizer.minimize(loss)

# 执行训练
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for step in range(d_size+1):
        sess.run(train)
        if step % 20 == 0:
            print('步数：',step,'；参数：',sess.run(W),'',sess.run(b))
#    print(y_test.eval())
