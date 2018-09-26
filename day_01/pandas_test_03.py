# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 10:37:31 2018

@author: LZC
"""

from __future__ import print_function

# 导入 pandas API 并输出了相应的 API 版本
import pandas as pd
print('当前pandas版本号：',pd.__version__)

'''
pandas 中的主要数据结构被实现为以下两类：

DataFrame，您可以将它想象成一个关系型数据表格，其中包含多个行和已命名的列。
Series，它是单一列。DataFrame 中包含一个或多个 Series，每个 Series 均有一个名称。

数据框架是用于数据操控的一种常用抽象实现形式。
'''
# 创建 Series 的一种方法是构建 Series 对象，方法如下：
city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento']) # 习题例子
data_A = pd.Series(['AAA','BBB','FFF','EEE','PPP'])# 新建Series

print('例子数据城市名：')
print(city_names)
print('数据A：')
print(data_A)

# 您可以将映射 string 列名称的 dict 传递到它们各自的 Series，从而创建DataFrame对象。
# 如果 Series 在长度上不一致，系统会用特殊的 NA/NaN 值填充缺失的值。
population = pd.Series([852469, 1015785, 485199]) # 增加数据
data_B = pd.Series(['word1','word2','word3']) # 数据个数可以不同
Data = pd.Series([293184,3184193,123421,1343321])

cities = pd.DataFrame({ 'City name': city_names, 'Population': population })# 建立DataFrame
data_C = pd.DataFrame({'index':data_B,'name':data_A,'data':Data})

print('\n\b现有城市数据如下：')
print(cities)
print('现有自创数据如下：')
print(data_C)

# 可以使用熟悉的 Python dict/list 指令访问 DataFrame 数据
print('\n\bSeries中数据类型：')
print(type(cities['City name']))
print('可依据key访问某列：')
print(cities['City name'])
print('\n\bSeries中某元素类型：')
print(type(cities['City name'][1]))
print('可依据索引访问某元素：')
print(cities['City name'][1])
print('\n\bDataFrame同样可以使用索引,但不可单独使用索引引用')
print(type(cities[0:3]))
print(cities[0:3])

print('\n\b可进行数学计算')
import numpy as np
print('人口整体除1000:')
print(population/1000)
print('人口整体取对数:')
print(np.log(population))

# 下面的示例创建了一个指明 population 是否超过 100 万的新 Series
# 对于更复杂的单列转换，您可以使用 Series.apply。像 Python 映射函数一样，
# Series.apply 将以参数形式接受 lambda 函数，而该函数会应用于每个值。
population_large = population.apply(lambda val: val > 1000000)
# DataFrames 的修改方式也非常简单。例如，以下代码向现有 DataFrame 添加了三个 Series
cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
cities['population more than 1000000'] = population_large
cities['Population density'] = cities['Population'] / cities['Area square miles']
print('\n\b更新后的DataFrame')
print(cities)
print('单独访问新列')
print(cities['population more than 1000000'])

print(cities.index)
print(city_names.index)
cities.reindex([2, 0, 1])
print(cities)
# 调用 DataFrame.reindex 以手动重新排列各行的顺序。例如，以下方式与按城市名称排序具有相同的效果：
cities.reindex(np.random.permutation(cities.index))
print(cities)