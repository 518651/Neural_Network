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


class Regressor(paddle.nn.Layer):
    # self代表类的实例自身
    def __init__(self):
        # 初始化父类的一些参数
        super(Regressor, self).__init__()

        # 定义一层全连接层， 输入维度是13， 输出维度是1
        self.fc = Linear(in_features=13, out_features=1)

    def forward(self, inputs):
        x = self.fc(inputs)  # 调用全连接层函数进行向前计算
        return x


# 声明定义好的线性回归模型
model = Regressor()
# 开启模型训练模式
model.train()
# 加载数据
training_data, test_data = load_data()
# 定义优化算法，使用随机梯度下降SGD
# 学习率设置为0.01
opt = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())

EPOCH_NUM = 10
BATCH_SIZE = 10

for epoch_id in range(EPOCH_NUM):
    # 在每轮迭代开始之前，将训练数据的顺序随机的打乱
    np.random.shuffle(training_data)
    # 将训练数据进行拆分,每个batch包含10条数据
    mini_batches = [training_data[k:k + BATCH_SIZE] for k in range(0, len(training_data), BATCH_SIZE)]
    # 定义内层循环
    for iter_id, mini_batch in enumerate(mini_batches):
        x = np.array(mini_batch[:, :-1])  # 获取当前批次的x
        y = np.array(mini_batch[:, -1:])  # 获取当前批次的y
        # 将numpy数据转为Paddle动态图Tensor格式
        house_features = paddle.to_tensor(x)
        prices = paddle.to_tensor(y)

        # 向前计算
        predicts = model(house_features)

        # 计算损失
        loss = F.square_error_cost(predicts, label=prices)
        avg_loss = paddle.mean(loss)
        if iter_id % 20 == 0:
            print("epoch:{}, iter:{}, loss is {}".format(epoch_id, iter_id, avg_loss.numpy()))

        # 反向传播，计算每次参数的梯度值
        avg_loss.backward()
        # 更新参数，根据设置好的学习率迭代一代
        opt.step()
        # 清空梯度变量,以备下一轮计算
        opt.clear_grad()


# 保存模型参数，文件名为LR_model.pdparams
# paddle.save(model.state_dict(), 'LR_model.pdparams')
# print("模型保存成功，模型参数保存在LR_model.pdparams中")

def load_one_example():
    # 从上边已加载的测试集中，随机选择一条作为测试数据
    idx = np.random.randint(0, test_data.shape[0])
    idx = -14
    one_data, label = test_data[idx, :-1], test_data[idx, -1]
    # 修改该条数据shape为[1,13]
    one_data = one_data.reshape([1, -1])

    return one_data, label


# 参数为保存模型参数的文件地址
model_dict = paddle.load('LR_model.pdparams')
model.load_dict(model_dict)
model.eval()

# 参数为数据集的文件地址
one_data, label = load_one_example()
# 将数据转为动态图的variable格式
one_data = paddle.to_tensor(one_data)
predict = model(one_data)

# 对结果做反归一化处理
predict = predict * (max_values[-1] - min_values[-1]) + min_values[-1]
# 对label数据做反归一化处理
label = label * (max_values[-1] - min_values[-1]) + min_values[-1]

print("预测结果为 {}, 相应的标签是 {}".format(predict.numpy(), label))
