import paddle
from paddle.nn import Linear
import paddle.nn.functional as F
import os
import gzip
import json
import random
import numpy as np


def load_data(mode='train'):
    datafile = './work/mnist.json.gz'
    print('loading mnist dataset from {}'.format(datafile))
    data = json.load(gzip.open(datafile))
    print('mnist dataset load done')

    train_set, val_set,eval_set = data
    if mode == 'train':
        imgs, labels = train_set[0], train_set[1]
    elif mode == 'valid':
        imgs, labels = val_set[0], val_set[1]
    elif mode == 'eval':
        imgs , labels = eval_set[0], eval_set[1]
    else:
        raise Exception('mode can only be one of [\'train\',\'valid\',\'eval\']')

    print('训练集数量:', len(imgs))

    # 校验数据
    imgs_length = len(imgs)

    assert len(imgs) == len(labels), \
    'length of train_imgs({}) should be the same as train_labels({})'.format(len(imgs),len(labels))

    # 获得数据集长度
    imgs_length = len(imgs)

    # 定义数据集每个数据的序号, 根据序号读取数据
    index_list = list(range(imgs_length))

    BATCHSIZE = 100

    # 定义数据生成器
    def data_generator():
        if mode == 'train':
            random.shuffle(index_list)
        imgs_list = []
        labels_list = []
        for i in index_list:
            img = np.array(imgs[i]).astype('float32')
            label = np.array(labels[i]).astype('float32')
            imgs_list.append(img)
            labels_list.append(label)
            if len(imgs_list) == BATCHSIZE:
                yield np.array(imgs_list), np.array(labels_list)
                imgs_list = []
                labels_list = []

        if len(imgs_list) > 0:
            yield np.array(imgs_list), np.array(labels_list)
    return data_generator


class MNIST(paddle.nn.Layer):
    def __init__(self):
        super(MNIST, self).__init__()

        #定义一层全连接层
        self.fc = paddle.nn.Linear(in_features=784, out_features=1)

    def forward(self, inputs):
        outputs = self.fc(inputs)
        return outputs


def train(model):
    model = MNIST()
    model.train()

    train_loader = load_data('train')
    opt = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())
    EPOCH_NUM = 10
    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            images, labels = data
            images = paddle.to_tensor(images)
            labels = paddle.to_tensor(labels)

            # 向前计算
            predits = model(images)

            # 计算loss
            loss = F.square_error_cost(predits, labels)
            avg_loss = paddle.mean(loss)

            if batch_id % 200 ==0:
                print('epoch:{},batch:{},loss is {}'.format(epoch_id,batch_id, avg_loss))

            # 向后传播
            avg_loss.backward()
            opt.step()
            opt.clear_grad()

    # 保存模型
    paddle.save(model.state_dict(), './mnist.pdparams')
# 创建模型
model = MNIST()
train(model)


class MnistDataset(paddle.io.Dataset):
    def __init__(self, mode):
        datafile = './work/mnist.json.gz'
        data = json.load(gzip.open(datafile))

        train_set, val_set, eval_set = data
        if mode=='train':
            imgs, labels = train_set[0], train_set[1]

        elif mode == 'valid':
            imgs, labels = val_set[0], val_set[1]

        elif mode == 'eval':
            imgs, labels = eval_set[0], eval_set[1]

        else:
            raise Exception('mode can only be one of [train, valid, eval]')

        imgs_length = len(imgs)
        assert len(imgs) == len(labels), \
            'length of train_imgs({}) should be the same as train_labels({})'.format(len(imgs), len(labels))

        self.imgs = imgs
        self.labels = labels

    def __getitem__(self, idx):
        img = np.array(self.imgs[idx]).astype('float32')
        labels = np.array(self.labels[idx]).astype('float32')

        return img, labels

    def __len__(self):
        return len(self.imgs)


#声明数据加载函数,使用mnistDataset数据集
train_dataset = MnistDataset(mode='train')
#使用paddle.io.Dataloader 定义DataLoader对象用于加载python生成器产生的数据
# Dataloader 返回一个批次数据迭代器,并且是异步
data_loader = paddle.io.DataLoader(train_dataset, batch_size=100, shuffle=True)
#迭代的读取数据并打印数据形状
for i, data in enumerate(data_loader()):
    images, labels = data
    print(i, images.shape, labels.shape)
    if i>=2:
        break
