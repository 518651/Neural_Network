import paddle
from paddle.nn import Linear
import paddle.nn.functional as F
import numpy as np
import os
import random


def load_data():
    datafile = './work/housing.data'
    data = np.fromfile(datafile, sep=' ', dtype=np.float32)

    # 每条数据包括14项, 其中前13项是x,第14项为y
    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT',
                     'MEDV']
    feature_num = len(feature_names)
    print("获取到的featture数量为:", feature_num)

    # 将原始数据进行Reshape，变成[N, 14]这样的形状
    data = data.reshape([data.shape[0] // feature_num, feature_num])
    print("数据集Tensor共有(列):", data.shape[0] // feature_num)

    # 将原数据集拆分成训练集和测试集
    # 8:2 训练集 | 测试集
    # 训练集和测试集没有交集
    ratio = 0.8
    offset = int(data.shape[0] * ratio)  # 506 * 0.8 = 404 个训练集
    training_data = data[:offset]

    # 计算train数据集的最大、小值
    maximums, minimums = training_data.max(axis=0), training_data.min(axis=0)

    # 数据归一化
    global max_values
    global min_values

    max_values = maximums
    min_values = minimums
    for i in range(feature_num):
        data[:, i] = (data[:, i] - min_values[i]) / (maximums[i] - minimums[i])

    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data


training_data, test_data = load_data()
print(training_data.shape)
print(training_data[1, :])
