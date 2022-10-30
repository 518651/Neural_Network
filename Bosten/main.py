import numpy as np
import json

from matplotlib import pyplot as plt


def loader_data():
    datafile = './work/housing.data'
    data = np.fromfile(datafile, sep=' ')
    print(data)

    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                     'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    feature_num = len(feature_names)  # 获取feature_names 的长度
    data = data.reshape([data.shape[0] // feature_num, feature_num])  # 转化为 N * 14的形式

    x = data[0]
    print(x.shape)
    print(x)

    # 对训练集进行分类
    radio = 0.8
    offset = int(data.shape[0] * radio)
    print(offset)
    training_data = data[:offset]
    print(training_data.shape)

    # 数据归一化处理
    # 计算train数据集的最大值，最小值，平均值
    maximums, minimums, avgs = \
        training_data.max(axis=0), \
        training_data.min(axis=0), \
        training_data.sum(axis=0) / training_data.shape[0]

    # 对数据进行归一化处理
    for i in range(feature_num):
        print(maximums[i], minimums[i], avgs[i])
        data[:, i] = (data[:, i] - minimums[i]) / (maximums[i] - minimums[i])

    # 训练集和测试集的划分比例
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data


# 获取数据
training_data, test_data = loader_data()
x = training_data[:, :-1]
y = training_data[:, -1:]

# 查看训练集数据
print("---------")
print(x.shape)
print(y.shape)
print(x[0])
print(y[0])

# 模型设计
# 实现模型“前向计算”（从输入到输出）的过程
# 输入特征x有13个分量，y有1个分量，那么参数权重的形状（shape）是13×1
# 0~1之前任意数字初始化
w = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, -0.1, -0.2, -0.3, -0.4, 0.0]
w = np.array(w).reshape([13, 1])  # 用numpy创建一个13 * 1的数组


class Network(object):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，此处设置固定的随机数种子
        # np.random.seed(0)
        self.w = np.random.randn(num_of_weights, 1)
        self.b = 0.

    def forward(self, x):
        z = np.dot(x, self.w) + self.b  # 向量点积
        return z

    def loss(self, z, y):
        error = z - y
        num_samples = error.shape[0]
        cost = error * error
        cost = np.sum(cost) / num_samples
        return cost

    def gradient(self, x, y):
        z = self.forward(x)
        N = x.shape[0]
        gradient_w = 1. / N * np.sum((z - y) * x, axis=0)  # 对行求平均值
        gradient_w = gradient_w[:, np.newaxis]  # 不符合模型输入要求，因此要单独加1个维度|表示数据不变，单独加一个维度
        gradient_b = 1. / N * np.sum(z - y)
        return gradient_w, gradient_b

    def update(self, gradient_w, gradient_b, eta=0.01):
        self.w = self.w - eta * gradient_w
        self.b = self.b - eta * gradient_b
    # eta 学习率| 步长 每次运算下降多少由学习率决定
    # 学习率设置过大，可能会错过正确值。
    # 学习率设置过小，参数更新很慢

    def train(self, training_data, num_epochs, batch_size=10, eta=0.01):
        n = len(training_data)
        losses = []
        for epoch_id in range(num_epochs):
            # 外层循环表示迭代的轮数
            # 在每轮迭代开始之前，将训练数据的顺序随机打乱
            # 然后再按每次取batch_size条数据的方式取出
            np.random.shuffle(training_data)
            # 将训练数据进行拆分，每个mini_batch包含batch_size条的数据
            mini_batches = [training_data[k:k + batch_size] for k in range(0, n, batch_size)]
            # mini_batches为一个列表，列表里的每一个元素均为，batch_size行14列的矩阵
            for iter_id, mini_batch in enumerate(mini_batches):
                # 每次进入网络batch_size行14列的矩阵，直到一个epoch结束。即所有数据均被使用一次
                print(self.w.shape)
                print(self.b)
                x = mini_batch[:, :-1]
                y = mini_batch[:, -1:]
                a = self.forward(x)
                loss = self.loss(a, y)
                gradient_w, gradient_b = self.gradient(x, y)
                self.update(gradient_w, gradient_b, eta)
                losses.append(loss)
                print('Epoch {:3d} / iter {:3d}, loss = {:.4f}'.
                      format(epoch_id, iter_id, loss))

        return losses


# 获取数据
train_data, test_data = loader_data()

# 创建网络
net = Network(13)
# 启动训练
losses = net.train(train_data,
                   num_epochs=500,  # num_epochs 表示迭代的轮数
                   batch_size=100,  # batch_size 表示一个batch所包含样本数据的大小
                   eta=0.1)

# 画出损失函数的变化趋势
plot_x = np.arange(len(losses))
plot_y = np.array(losses)
plt.plot(plot_x, plot_y)
plt.show()
