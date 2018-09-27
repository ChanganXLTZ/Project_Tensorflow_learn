# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 15:19:16 2018

@author: LZC
"""
from __future__ import print_function
import tensorflow as tf

#创建和操控两个矢量（一维张量），每个矢量正好六个元素：
g1 = tf.Graph()
with g1.as_default():
    # Create a six-element vector (1-D tensor).
    primes = tf.constant([2, 3, 5, 7, 11, 13], dtype=tf.int32,name='const1')

    # Create another six-element vector. Each element in the vector will be
    # initialized to 1. The first argument is the shape of the tensor (more
    # on shapes below).
    n = primes.shape # 读取primes中数组长度，使用n保存
    m = primes.get_shape() # 另一种读取长度的方法，调用函数读取
    ones = tf.ones([n[0]], dtype=tf.int32)
    
    # Add the two vectors. The resulting tensor is a six-element vector.
    just_beyond_primes = tf.add(primes, ones)
    # m = just_beyond_primes.
    # Create a session to run the default graph.
    with tf.Session() as sess:
        print('\n\b=====计算图-1=====')
        print('原始数据长度：',n[0])
        print('原始数据长度：',m[0])
        print('原始数据：')
        print(sess.run(primes))
        print('变量名：',primes.name)
        print(ones.dtype)
        print('加1：')
        print(just_beyond_primes.eval())
'''   
形状用于描述张量维度的大小和数量。张量的形状表示为列表，其中第 i 个元素表示维度 i 的大小。
列表的长度表示张量的阶（即维数）。有关详情，请参阅 TensorFlow 文档。
以下是一些基本示例：
'''

g2 = tf.Graph()
with g2.as_default():
    # A scalar (0-D tensor).
    scalar1 = tf.zeros([]) # 单一标量
    scalar2 = tf.zeros([0]) # 空向量
    scalar3 = tf.zeros([1]) # 单一元素向量

    # A vector with 3 elements.
    vector = tf.zeros([3])

    # A matrix with 2 rows and 3 columns.
    matrix = tf.zeros([2, 3])

    with tf.Session() as sess:
        print('\n\b=====计算图-2=====')
        print('这个张量阶数：', scalar1.shape, 'and value:', scalar1.eval())
        print('这个向量阶数：', scalar2.get_shape(), 'and value:', scalar2.eval())
        print('这个向量阶数：', scalar3.get_shape(), 'and value:', scalar3.eval())
        print('这个向量阶数：', vector.get_shape(), 'and value:', vector.eval())
        print('这个矩阵阶数：', matrix.get_shape(), 'and value:\n', matrix.eval())

# 广播功能，对阶数低的张量进行扩张
g3 = tf.Graph()
with g3.as_default():
    # Create a six-element vector (1-D tensor).
    primes = tf.constant([2, 3, 5, 7, 11, 13], dtype=tf.int32)

    # Create a constant scalar with value 1.
    ones = tf.constant(1, dtype=tf.int32)

    # Add the two tensors. The resulting tensor is a six-element vector.
    just_beyond_primes = tf.add(primes, ones)

    with tf.Session() as sess:
        print('\n\b=====计算图-3=====，此处使用广播功能：')
        print('被加张量长度：',ones.get_shape())
        print('被加张量的值',sess.run(ones))
        print(just_beyond_primes.eval())
        
# 矩阵乘法
g4 = tf.Graph()
with g4.as_default():
    # Create a matrix (2-d tensor) with 3 rows and 4 columns.
    x = tf.constant([[5, 2, 4, 3], [5, 1, 6, -2], [-1, 3, -1, -2]],
                    dtype=tf.int32)

    # Create a matrix with 4 rows and 2 columns.
    y = tf.constant([[2, 2], [3, 5], [4, 5], [1, 6]], 
                    dtype=tf.int32)

    # Multiply `x` by `y`. 
    # The resulting matrix will have 3 rows and 2 columns.
    matrix_multiply_result = tf.matmul(x, y)

    with tf.Session() as sess:
        print('\n\b=====计算图-4=====')
        print('矩阵1：\n',sess.run(x))
        print('矩阵2：\n',sess.run(y))
        print('矩阵相乘结果：\n',matrix_multiply_result.eval())
    
# 张量变形
# 由于张量加法和矩阵乘法均对运算数施加了限制条件，
# TensorFlow 编程者肯定会频繁改变张量的形状。
# 您可以使用 tf.reshape 方法改变张量的形状。 
# 例如，您可以将 8x2 张量变形为 2x8 张量或 4x4 张量：

g5 = tf.Graph()
with g5.as_default():
    # Create an 8x2 matrix (2-D tensor).
    matrix = tf.constant([[1,2], [3,4], [5,6], [7,8],
                          [9,10], [11,12], [13, 14], [15,16]],
        dtype=tf.int32)

    # Reshape the 8x2 matrix into a 2x8 matrix.
    reshaped_2x8_matrix = tf.reshape(matrix, [2,8])

    # Reshape the 8x2 matrix into a 4x4 matrix.
    reshaped_8x2_matrix = tf.reshape(reshaped_2x8_matrix, [8,2])

    with tf.Session() as sess:
        print('\n\b=====计算图-5=====')
        print("Original matrix (8x2):")
        print(sess.run(matrix))
        print("Reshaped matrix (2x8):")
        print(reshaped_2x8_matrix.eval())
        print("Reshaped matrix (8x2):")
        print(reshaped_8x2_matrix.eval())
        
# 此外，您还可以使用 tf.reshape 更改张量的维数（\'阶\'）。 
# 例如，您可以将 8x2 张量变形为三维 2x2x4 张量或一维 16 元素张量。
g6 = tf.Graph()
with g6.as_default():
    # Create an 8x2 matrix (2-D tensor).
    matrix = tf.constant([[1,2], [3,4], [5,6], [7,8],
                          [9,10], [11,12], [13, 14], [15,16]], 
        dtype=tf.int32)

    # Reshape the 8x2 matrix into a 3-D 2x2x4 tensor.
    reshaped_2x2x4_tensor = tf.reshape(matrix, [8,2,1])
  
    # Reshape the 8x2 matrix into a 1-D 16-element tensor.
    one_dimensional_vector = tf.reshape(matrix, [16])

    with tf.Session() as sess:
        print('\n\b=====计算图-6=====')
        print("Original matrix (8x2):")
        print(matrix.eval())
        print("Reshaped 3-D tensor (2x2x4):")
        print(reshaped_2x2x4_tensor.eval())
        print("1-D vector:")
        print(one_dimensional_vector.eval())

