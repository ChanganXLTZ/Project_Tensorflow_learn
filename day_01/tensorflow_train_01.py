# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 15:17:04 2018

@author: LZC
"""
## 加载必要的库。
from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

## 加载数据集。
california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv"
                                           , sep=",")
''' 
对数据进行随机化处理，以确保不会出现任何病态排序结果（可能会损害随机梯度下降法的效果）。
此外，我们会将 median_house_value 调整为以千为单位，这样，模型就能够以常用范围内的
学习速率较为轻松地学习这些数据。
'''
california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe["median_house_value"] /= 1000.0 # 将元素除以1000，缩小范围
'''
建议您在使用数据之前，先对它有一个初步的了解。
输出关于各列的一些实用统计信息快速摘要：样本数、均值、标准偏差、最大值、最小值和各种分位数。
'''
print(california_housing_dataframe.describe())

# =====第 1 步：定义特征并配置特征列=====
'''
在 TensorFlow 中，我们使用一种称为“特征列”的结构来表示特征的数据类型。
特征列仅存储对特征数据的描述；不包含特征数据本身。
一开始，我们只使用一个数值输入特征 total_rooms。以下代码会从 california_housing_dataframe中
提取 total_rooms 数据，并使用 numeric_column 定义特征列，这样会将其数据指定为数值：
'''
# Define the input feature: total_rooms.
my_feature = california_housing_dataframe[["total_rooms"]]
# Configure a numeric feature column for total_rooms.
feature_columns = [tf.feature_column.numeric_column("total_rooms")]
# total_rooms 数据的形状是一维数组（每个街区的房间总数列表）。
# 这是 numeric_column 的默认形状，因此我们不必将其作为参数传递。

# =====第 2 步：定义目标=====
#我们将定义目标，也就是 median_house_value。同样，我们可以从california_housing_dataframe中提取：
# Define the label.
targets = california_housing_dataframe["median_house_value"]

# =====第 3 步：配置 LinearRegressor=====
'''
接下来，我们将使用 LinearRegressor 配置线性回归模型，并使用 GradientDescentOptimizer
（它会实现小批量随机梯度下降法 (SGD)）训练该模型。learning_rate 参数可控制梯度步长的大小。
注意：为了安全起见，我们还会通过 clip_gradients_by_norm 将梯度裁剪应用到我们的优化器。
梯度裁剪可确保梯度大小在训练期间不会变得过大，梯度过大会导致梯度下降法失败。
'''
# Use gradient descent as the optimizer for training the model.
my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.00001)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

# Configure the linear regression model with our feature columns and optimizer.
# Set a learning rate of 0.0000001 for Gradient Descent.
linear_regressor = tf.estimator.LinearRegressor(
    feature_columns=feature_columns,
    optimizer=my_optimizer
)

# =====第 4 步：定义输入函数=====
'''
要将加利福尼亚州住房数据导入 LinearRegressor，我们需要定义一个输入函数，
让它告诉 TensorFlow 如何对数据进行预处理，以及在模型训练期间如何批处理、随机处理和重复数据。
首先，我们将 Pandas 特征数据转换成 NumPy 数组字典。然后，我们可以使用 
TensorFlow Dataset API 根据我们的数据构建 Dataset 对象，并将数据拆分成大小为 
batch_size 的多批数据，以按照指定周期数 (num_epochs) 进行重复。

注意：如果将默认值 num_epochs=None 传递到 repeat()，输入数据会无限期重复。
然后，如果 shuffle 设置为 True，则我们会对数据进行随机处理，以便数据在训练期间
以随机方式传递到模型。buffer_size 参数会指定 shuffle 将从中随机抽样的数据集的大小。
最后，输入函数会为该数据集构建一个迭代器，并向 LinearRegressor 返回下一批数据。
'''
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of one feature.
  
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. 
                  None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
  
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}                                           
 
    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(buffer_size=10000)
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

# =====第 5 步：训练模型=====
'''
现在，我们可以在 linear_regressor 上调用 train() 来训练模型。我们会将 my_input_fn 封装在
lambda 中，以便可以将 my_feature 和 target 作为参数传入，首先，我们会训练 100 步。
（有关详情，请参阅此 TensorFlow 输入函数教程）
'''
_ = linear_regressor.train(
    input_fn = lambda:my_input_fn(my_feature, targets),
    steps=500
)

# =====第 6 步：评估模型=====
'''
我们基于该训练数据做一次预测，看看我们的模型在训练期间与这些数据的拟合情况。
注意：训练误差可以衡量您的模型与训练数据的拟合情况，但并_不能_衡量模型泛化到新数据的效果。
在后面的练习中，您将探索如何拆分数据以评估模型的泛化能力。
'''
# Create an input function for predictions.
# Note: Since we're making just one prediction for each example, we don't 
# need to repeat or shuffle the data here.
prediction_input_fn =lambda: my_input_fn(my_feature, targets, num_epochs=1, shuffle=False)

# Call predict() on the linear_regressor to make predictions.
predictions = linear_regressor.predict(input_fn=prediction_input_fn)

# Format predictions as a NumPy array, so we can calculate error metrics.
predictions = np.array([item['predictions'][0] for item in predictions])

# Print Mean Squared Error and Root Mean Squared Error.
mean_squared_error = metrics.mean_squared_error(predictions, targets)
root_mean_squared_error = math.sqrt(mean_squared_error)
print("Mean Squared Error (on training data): %0.3f" % mean_squared_error)
print("Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error)
'''
由于均方误差 (MSE) 很难解读，因此我们经常查看的是均方根误差 (RMSE)。
RMSE 的一个很好的特性是，它可以在与原目标相同的规模下解读。

我们来比较一下 RMSE 与目标最大值和最小值的差值：
'''
min_house_value = california_housing_dataframe["median_house_value"].min()
max_house_value = california_housing_dataframe["median_house_value"].max()
min_max_difference = max_house_value - min_house_value

print("Min. Median House Value: %0.3f" % min_house_value)
print("Max. Median House Value: %0.3f" % max_house_value)
print("Difference between Min. and Max.: %0.3f" % min_max_difference)
print("Root Mean Squared Error: %0.3f" % root_mean_squared_error)

'''
这是每个模型开发者都会烦恼的问题。我们来制定一些基本策略，以降低模型误差。

首先，我们可以了解一下根据总体摘要统计信息，预测和目标的符合情况。
'''
calibration_data = pd.DataFrame()
calibration_data["predictions"] = pd.Series(predictions)
calibration_data["targets"] = pd.Series(targets)
calibration_data.describe()

'''
好的，此信息也许有帮助。平均值与模型的 RMSE 相比情况如何？各种分位数呢？

我们还可以将数据和学到的线可视化。我们已经知道，单个特征的线性回归可绘制成
一条将输入 x 映射到输出 y 的线。
首先，我们将获得均匀分布的随机数据样本，以便绘制可辨的散点图。
'''
sample = california_housing_dataframe.sample(n=300)
'''
然后，我们根据模型的偏差项和特征权重绘制学到的线，并绘制散点图。该线会以红色显示。
'''
# Get the min and max total_rooms values.
x_0 = sample["total_rooms"].min()
x_1 = sample["total_rooms"].max()

# Retrieve the final weight and bias generated during training.
weight = linear_regressor.get_variable_value('linear/linear_model/total_rooms/weights')[0]
bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

# Get the predicted median_house_values for the min and max total_rooms values.
y_0 = weight * x_0 + bias 
y_1 = weight * x_1 + bias

# Plot our regression line from (x_0, y_0) to (x_1, y_1).
plt.plot([x_0, x_1], [y_0, y_1], c='r')

# Label the graph axes.
plt.ylabel("median_house_value")
plt.xlabel("total_rooms")

# Plot a scatter plot from our data sample.
plt.scatter(sample["total_rooms"], sample["median_house_value"])

# Display graph.
plt.show()
'''
这条初始线看起来与目标相差很大。看看您能否回想起摘要统计信息，并看到其中蕴含的相同信息。

综上所述，这些初始健全性检查提示我们也许可以找到更好的线。
'''