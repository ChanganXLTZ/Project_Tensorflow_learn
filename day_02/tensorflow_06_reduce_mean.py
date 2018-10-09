# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 19:17:30 2018
本文将讲述reduce_mean()的使用方法；
tf.reduce_mean(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None)
根据给出的axis在input_tensor上求平均值。除非keep_dims为真，axis中的每个的张量秩会减少1。
如果keep_dims为真，求平均值的维度的长度都会保持为1。
如果不设置axis，所有维度上的元素都会被求平均值，并且只会返回一个只有一个元素的张量。
@author: 鱼香土豆丝
全文地址请点击：https://blog.csdn.net/he_min/article/details/78694383?utm_source=copy 
"""
import numpy as np
import tensorflow as tf

x = np.array([[1.,2.,3.],[4.,5.,6.]])
with tf.Session() as sess:
    mean_none = sess.run(tf.reduce_mean(x ,axis= None))
    mean_0 = sess.run(tf.reduce_mean(x, axis=0))
    mean_1 = sess.run(tf.reduce_mean(x, axis=1))
    print ('原数据：')
    print(x)
    print ('无参数直接调用：',mean_none,'——整体求平均')
    print ('第二形参带入0：',mean_0,'——按列求平均，每列求一个均值')
    print ('第二形参带入1：',mean_1,'——按行求平均，每行求一个均值')