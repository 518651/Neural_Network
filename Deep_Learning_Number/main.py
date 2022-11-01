import paddle
from paddle.nn import Linear
import paddle.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# 图像归一化函数，将数据范围为[0, 255]的图像归一化到[0, 1]
def norm_img(img):
    # 验证传入数据格式是否正确，img的shape为[batch_size, 28, 28]
    assert len(img.shape) == 3
    batch_size, img_h, img_w = img.shape[0], img.shape[1], img.shape[2]
    # 归一化图像数据
    img = img / 255
    # 将图像形式reshape为[batch_size, 784]
    img = paddle.reshape(img, [batch_size, img_h * img_w])

    return img


paddle.vision.set_image_backend('cv2')


class MNISTNET(paddle.nn.Layer):
    def __init__(self):
        super(MNISTNET, self).__init__()

        # 定义全连接层 , 输出维度是1
        self.fc = paddle.nn.Linear(in_features=784, out_features=1)

    # 定义向前计算
    def forward(self, inputs):
        outputs = self.fc(inputs)
        return outputs


# 实例化网络结构
model = MNISTNET()


def train(model):
    # 设置为训练模式
    model.train()

    # 加载训练集batch_size设置为 16
    train_loader = paddle.io.DataLoader(paddle.vision.datasets.MNIST(mode='train'),
                                        batch_size=16,
                                        shuffle=True)

    # 定义优化器
    opt = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())
    EPOCH_NUM = 10
    for epoch in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            images = norm_img(data[0]).astype('float32')
            labels = data[1].astype('float32')

            # 前向计算的过程
            predicts = model(images)

            # 计算损失
            loss = F.square_error_cost(predicts, labels)
            avg_loss = paddle.mean(loss)

            # 每训练100批次的数据,打印下当前loss的情况
            if batch_id % 1000 == 0:
                print("当前训练ID:{}, 批次ID:{}, 误差:{}".format(epoch, batch_id, avg_loss))

            avg_loss.backward()  # 反向传播，更新参数
            opt.step()
            opt.clear_grad()  # 清空梯度数据


train(model)
paddle.save(model.state_dict(), './mnist.pdparams')

image_path = './work/example_0.jpg'
# 读取图像并显示
im = Image.open('./work/example_0.jpg')
plt.imshow(im)

# 将原始图像转灰度图
im = im.convert('L')
print('原始图像shape', np.array(im).shape)
# 使用Image.ANTIALIAS方式采样原始图片
im = im.resize((28, 28), Image.ANTIALIAS)
# plt.show(im)
print('采样后图片shape', np.array(im).shape)


# 读取本地图片 转换成模型输入格式
def load_image(img_path):
    # 从img_path中读取图像,并转为灰度图
    im = Image.open(img_path).convert('L')
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).reshape(1, -1).astype(np.float32)
    im = 1 - im / 255
    return im


model = MNISTNET()
params_file_path = 'mnist.pdparams'
img_path = './work/example_0.jpg'
# 加载模型参数
param_dict = paddle.load(params_file_path)
model.load_dict(param_dict)
# 载入数据
model.eval()
tensor_img = load_image(img_path)
result = model(paddle.to_tensor(tensor_img))
print('result', result)
print('预测结果为:', result.numpy().astype('int32'))
plt.show()
