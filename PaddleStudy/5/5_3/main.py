import os
import random
import paddle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import gzip
import json

# 定义数据读取器
def loader_data(mode='train'):
    # Loading data
    datafile = './work/mnist.json.gz'
    print("loading mnist dataset from{}.........".format(datafile))
    data = json.load(gzip.open(datafile))
    print("mnist dataset load done")

    # 读取到的数据区分训练集,验证集,测试集
    train_set,val_set,eval_set = data

    # 数据集相关参数,图片高度IMG_ROWS,图片宽度IMG_COLS
    