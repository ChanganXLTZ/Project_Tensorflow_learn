# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 15:04:26 2018

@author: LZC
"""
import pandas as pd
import numpy as np
# print(pandas.__version__) # 与下句具有相同功能
pd.__version__

A = pd.Series(['AAA','BBB','FFF','EEE','PPP'])# 新建Series
print('数据A：') # 显示
print(A)
B = pd.Series(['word1','word2','word3']) # 数据个数可以不同
Data = pd.Series([293184,3184193,123421,1343321])

C = pd.DataFrame({'index':B,'vaule':A,'data':Data})# 建立DataFrame
print('DataFrame正确语法：')
print(C)

D = pd.DataFrame(B,A)# ？？？
print('DataFrame错误语法：')
print(D)

print(type(C['index'])) # 调用其他函数
print(C['index'])
print(np.log(C['data'])) # 调用其他包的函数
C['otherd'] = ['34','31','43','213','241'] # 添加元素
print(C)