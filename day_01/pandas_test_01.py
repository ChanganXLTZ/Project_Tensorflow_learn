# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 11:15:17 2018

@author: LZC
"""
# 本文件来源已不明确，大概目的为读取数据，生成描述
import pandas as pd

california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", 
                                           sep=",")
describtion = california_housing_dataframe.describe()
print('数据基本统计特性如下：')
print(describtion)
data_head = california_housing_dataframe.head()
print('数据举例：')
print(data_head)
data_index = california_housing_dataframe.keys()
n_data_index = len(data_index)
print('该数据具有如下特征：')
for i in range(n_data_index-1): 
    print(data_index[i],',',end='')
    # 注意这里range()方法返回一个左闭右开范围
print(data_index[n_data_index-1])
# 以下语句执行绘图，输入为数据一个特征，或称为 key
california_housing_dataframe.hist('housing_median_age')
california_housing_dataframe.hist(data_index[-2])