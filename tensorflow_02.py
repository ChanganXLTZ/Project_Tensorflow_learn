# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 13:59:57 2018

@author: LZC
"""
from __future__ import print_function
import tensorflow as tf
import matplotlib.pyplot as plt # 数据集可视化。
import numpy as np              # 低级数字 Python 库。
import pandas as pd             # 较高级别的数字 Python 库。

con_A = tf.constant([5,10,15,20])
con_B = tf.constant(('500','1000','1500','2000'))

with tf.Session() as Sess:
    print(Sess.run(con_A))
    print(Sess.run(con_B))

g1 = tf.Graph()
with g1.as_default():
    vir_A = tf.Variable([5,10,15,20])
    vir_B = tf.Variable([0,0,0,0])
    vir_B = vir_B.assign([500,1000,1500,2000])
       
    with tf.Session() as Sess:
        initialization = tf.global_variables_initializer()
        #print(Sess.run(vir_A)) # 直接执行报错
        
# 自定义 图
# Create a graph.
g2 = tf.Graph()

# Establish the graph as the "default" graph.
with g2.as_default():
  # Assemble a graph consisting of the following three operations:
  #   * Two tf.constant operations to create the operands.
  #   * One tf.add operation to add the two operands.
  x = tf.constant(8, name="x_const")
  y = tf.constant(5, name="y_const")
  z = tf.constant(4, name='z_const')
  sum = tf.add(x, y, name="x_y_sum")
  sum = tf.add(sum,z,name="x_y_z_sum")
  
  # Now create a session.
  # The session will run the default graph.
  with tf.Session() as sess:
    print(sum.eval())