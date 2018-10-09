# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 11:04:14 2018

@author: LZC
"""
import tensorflow as tf

x = tf.constant([2.0, 1.0])
y = tf.constant([1.0, 2.0])
z = x * y + x * x

dx, dy = tf.gradients(z, [x, y])  # 求z关于x,y的导数

# =============================================================================
# dx = tf.gradients(z,x)
# dy = tf.gradients(z,y)
# 出现：[array([5., 4.], dtype=float32)]的结果
# =============================================================================

with tf.Session() as sess:
    dx_v, dy_v = sess.run([dx, dy])
    print(dx_v)  # [5.0, 4.0]
    print(dy_v)  # [2.0, 1.0]
    print('分别得到z对x的偏导和z对y的偏导，并求出所在点的偏导值')