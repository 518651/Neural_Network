# `dTorch.nn`构建网络#2

​		在`torch.nn`拥有Torch准备好的层,可以方便使用者调用构建网络.本次笔记记录:**卷积层**、**池化层**、**激活函数层**、**循环层**、**全连接层**

## 卷积层

​		卷积层看作是输入和卷积核之前的内积运算,是两个实值函数之前的一种数学运算.在卷积运算中,通常使用卷积核将输入数据进行卷积运算得到输出作为特征映射,每个卷积核可获得一个特征映射.                           

​        <img src="F:\PenRoad\Neural NetWork\Pytorch快速入门\Pictrue\二维卷积运算过程.jpg" alt="二维卷积运算过程" style="zoom:25%;" />

​									**如上图二维卷积运算过程示例可以看出,卷积操作将周围几个像素的取值经过计算得到一个像素值.**

​		使用卷积运算,在图像识别、图像分割、图像重建等应用有三个好处,既卷积稀疏连接、参数共享、等变表示.（具体参数解释见:Page39）

​		在实际构建过程中,针对卷积操作的对象和使用场景不同,有一维卷积、二维卷积、三维卷积与转置卷积.均可以在`torch.nn`模块中调用

### 常用卷积操作对应类



|          层对应的类          |           功能作用           |
| :--------------------------: | :--------------------------: |
|     `torch.nn.Conv1d()`      |  针对输入信号上应用的1D卷积  |
|     `torch.nn.Conv2d()`      |  针对输入信号上应用的2D卷积  |
|     `torch.nn.Conv3d()`      |  针对输入信号上应用的3D卷积  |
| `torch.nn.ConvTranspose1d()` | 在输入信号上应用的1D转置卷积 |
| `torch.nn.ConvTranspose2d()` | 在输入信号上应用的2D转置卷积 |
| `torch.nn.ConvTranspose3d()` | 在输入信号上应用的3D转置卷积 |



### Use Explorer(`torch.nn.Conv2d`):

​	`torch.nn.Conv2d(in_channels,out_channels,kernel_size,stride=1.padding=0,dilation=1,groups=1,bias=True)`

​	参数解释:

​			***in_channels:(整数)输入图像的通道数***

​			***out_channels:(整数)经过卷积运算后,输出特征映射数量***

​			***kernel_size:(整数或数组)卷积核大小***

​			***stride:(整数或数组、正数)卷积的步长,默认为1.***

​			***padding:(整数或者数组，正数)在输入两边进行0的填充数量,默认为0***

​			***dilation:(整数或者数组,正数)卷积核元素之间的步幅,该参数可调整空洞卷积的大小,默认为1***

​			***groups:(整数,正数):冲输入通道到输出通道的阻塞连接数***

​			***bias(布尔值,正数):如果bias=True,则添加偏置,默认为True***

​			***torch.nn.Conv2d()输入Tensor为(N,C~in~,H~in~,W~in~),输出为(N,C~out~,H~out~,W~out~)***



### 对图像进行一次二维卷积

#### 		使用PIL包读取图像,使用matplotlib来可视化图像核卷积结果

```python
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image

# 读取一张图像
readimage = Image.open("data/Sample/Sample01.png")
# 转化为灰度图片 + 转化为Numpy数组
imagesetgray = np.array(readimage.convert("L"), dtype=np.float32) # 转换为 Numpy数组
# 可视化图片
plt.figure(figsize=(6, 6))
plt.imshow(imagesetgray, cmap=plt.cm.gray) # 转换为灰度图片
plt.axis("off")
plt.show()
```

输出:

<img src="F:\Project\Python\practice\pythonProject\data\Sample\Samplegray.png" alt="Samplegray" style="zoom:50%;" />

> ##### **在上面过程中,我们对测试图片进行了读取-->转换为Numpy数组(1080 * 1920)-->将图像转换为灰度图片-->显示图片的操作**

#### *现在开始对Numpy数组进行Terson转换,使Tensor能直接被Torch使用*

```python
# 开始将数组转化为Tensor
imh, imw = imagesetgray.shape
imagesetgray_t = torch.from_numpy(imagesetgray.reshape((1, 1, imh, imw)))
print("Numpy数组转换Tensor 为："+ str(imagesetgray_t.shape))
```

##### 输出：`Numpy数组转换Tensor 为：torch.Size([1, 1, 1080, 1920])`

对显示的灰度图进行卷积并提取图像轮廓

```python
# 对灰度图像进行卷积提取图像轮廓
kersize = 5  # 定义边缘检测卷积核,并将维度处理为1*1*5*5
ker = torch.ones(kersize, kersize, dtype=torch.float32) * -1
print("正在生成:" + str(ker.shape) + "的 全1 Tensor")
ker[2, 2] = 24
ker = ker.reshape((1, 1, kersize, kersize))
conv2d = nn.Conv2d(1, 2, (kersize, kersize), bias=False)
print("Conv2d卷积参数为:  " + str(conv2d))
conv2d.weight.data[0] = ker
imconv2dout = conv2d(imagesetgray_t)  # 向Conv2d输入待卷积数据并接收输出Tensor
print(imconv2dout)
imconv2dout_im = imconv2dout.data.squeeze()
print("经过卷积后端尺寸" + str(imconv2dout_im.shape))
plt.subplot(1, 2, 1)
plt.imshow(imconv2dout_im[0], cmap=plt.cm.gray) #使用边缘特征提取卷积核 | 很好的提取到图像边缘信息
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(imconv2dout_im[1], cmap=plt.cm.gray) #使用随机数卷积核 | 得到的卷积结果和原图很相似
plt.axis("off")
plt.show()
```

输出:`正在生成:torch.Size([5, 5])的 全1 Tensor`

​	`Conv2d卷积参数为:  Conv2d(1, 2, kernel_size=(5, 5), stride=(1, 1), bias=False)`

​	print(imconv2dout)因输出Tensor过于大,暂时不展示

​	`经过卷积后端尺寸torch.Size([2, 1076, 1916])`

<img src="F:\Project\Python\practice\pythonProject\data\Sample\Sample_conv2d.png" alt="Sample_conv2d" style="zoom:50%;" />



### 池化层



#### 池化层解释:

​		**池化操作的一个重要的目的是对卷积后得到的特征进行进一步处理(主要是降维),池化层可以起到对数据进一步浓缩的效果,从而缓解计算时内存压力.池化会选取一定大小区域,将该区域内的像素值使用一个元素表示.如果使用平均值代替,称为平均池化;如果使用最大值代替则称为最大值池化.**

<img src="F:\PenRoad\Neural NetWork\Pytorch快速入门\Pictrue\最大值池化和平均值池化.jpg" style="zoom:15%;" />

​		Pytorch中,提供了多种池化的类,分别是最大值池化(MaxPool)、最大值池化的逆过程(MaxUnPool),平均值池化(AvgPool)与自适应池化(AdaptiveMaxPool,AdaptiveAvgPool).并提供一维、二维和三维度的池化操作。



#### 池化函数:

|           层对应的类           |                 功能                 |
| :----------------------------: | :----------------------------------: |
|     `torch.nn.MaxPool1d()`     |    针对输入信号上应用1D最大池化值    |
|     `torch.nn.MaxPool2d()`     |    针对输入信号上应用2D最大池化值    |
|     `torch.nn.MaxPool3D()`     |    针对输入信号上应用3D最大池化值    |
|    `torch.nn.MaxUnPool1d()`    |       1D最大值池化的部分逆运算       |
|    `torch.nn.MaxUnPool2d()`    |       2D最大值池化的部分逆运算       |
|    `torch.nn.MaxunPool3d()`    |       3D最大值池化的部分逆运算       |
|     `torch.nn.AvgPool1d()`     |    针对输入信号上应用1D平均值池化    |
|     `torch.nn.AvgPool2d()`     |    针对输入信号上应用2D平均值池化    |
|     `torch.nn.AvgPool3d()`     |    针对输入信号上应用3D平均值池化    |
| `torch.nn.AdaptiveMaxPool1d()` | 针对输入信号上应用1D自适应最大值池化 |
| `torch.nn.AdaptiveMaxPool2d()` | 针对输入信号上应用2D自适应最大值池化 |
| `torch.nn.AdaptiveMaxPool3d()` | 针对输入信号上应用3D自适应最大值池化 |
| `torch.nn.AdaptiveAvgPool1d()` | 针对输入信号上应用1D自适应平均值池化 |
| `torch.nn.AdaptiveAvgPool2d()` | 针对输入信号上应用2D自适应平均值池化 |
| `torch.nn.AdaptiveAvgPool3d()` | 针对输入信号上应用3D自适应平均值池化 |

#### 		**对于`torch.nn.MaxPool2d()`池化操作相关参数,使用方法如下:**

```
torch.nn.MaxPool2d(kernel_size,stride=none,padding=0,dilation=1,return_indices=False,ceil_mode=Fales)
```

参数解释:

​	kernel_size:(整数或数组)最大值池化窗口大小

​	stride:(整数或数组,正数)最大值池化窗口移动布长,默认值是Kernel_size

​	padding:(整数或数组,正数)输入的每一条边补充0的层数

​	dilation:(整数或数组,正数)一个控制窗口中元素步幅的参数

​	return_indices:如果等于True,则会返回输出最大值索引,这样会更加便于之后的`torch.nn.MaxUnpool2d()`操作

​	ceil_mode:如果等于True,计算输出信号大小的时候,会使用向上取整,默认是向下取整.

​	**torch.nn.MaxPool2d()输入为(N,C~IN~,H~IN~,W~IN~)的Tensor,输出为(N,C~out~,H~out~,W~out~)的Terson**



##### Explorer-最大值池化:

```python
# 对边缘卷积结果进行最大值池化处理
MaxPool2 = nn.MaxPool2d(2, stride=2)
Poolout = MaxPool2(imconv2dout)
Pool_out_im = Poolout.squeeze()
# print(str(Poolout.shape))
```

可视化池化数据:

```python
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(Pool_out_im[0].data, cmap=plt.cm.gray)
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(Pool_out_im[1].data, cmap=plt.cm.gray)
plt.axis("off")
plt.show()
```

输出:

<img src="F:\Project\Python\practice\pythonProject\data\Sample\MaxPool2.png" style="zoom:50%;" />



平均值池化:

```python
# AvgPool2d() 平均值池化
avgpool2 = nn.AvgPool2d(2, stride=2)
avgpool2_out = avgpool2(imconv2dout)
avgpool2_out_im = avgpool2_out.squeeze()
print(avgpool2_out.shape)
```

输出:`torch.Size([1, 2, 538, 958])`



显示可视化数据:

```
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(avgpool2_out_im[0].data , cmap=plt.cm.gray)
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(avgpool2_out_im[1].data , cmap=plt.cm.gray)
plt.axis("off")
plt.show()
```

输出:

<img src="F:\Project\Python\practice\pythonProject\data\Sample\AvgPool.png" style="zoom:50%;" />



总结:

​		最大值池化:在池化前卷积输出维度尺寸为:`torch.Size([2, 1076, 1916])` 在经过窗口为 **2X2** ，步长为**2** 最大值池化后维度尺寸为:`torch.Size([1, 2, 538, 958])`

​		池化后的特征映射的尺寸越小,图片就越模糊

​		

## 激活层

### 		[激活函数](https://so.csdn.net/so/search?q=激活函数&spm=1001.2101.3001.7020)（activation functions）的目标是，**将神经网络非线性化。激活函数是连续的（continuous），且可导的（differential）**。

- 连续的：当输入值发生较小的改变时，输出值也发生较小的改变；
- 可导的：在定义域中，每一处都是存在导数；

**如果在神经元之间，没有使用非线性激活函数或者使用恒等激活函数，那么神经网络仅仅是将输入线性组合再输出。在这种情况下，多层神经网络与仅仅只有一层隐藏层的神经网络没有任何区别。因此，要想多个隐藏层有意义，必须使用非线性激活函数。**

==**激活函数基本介绍**==

==**神经网络是通过算法模拟生物的神经传递实现的算法。在生物的[神经元](https://so.csdn.net/so/search?q=神经元&spm=1001.2101.3001.7020)之间的传递依赖于神经元的激活状态，即点火与否，分别为1和0。为了模拟这种神经元传递，使用构建激活函数的方式进行信息传递。**==

<img src="https://img-blog.csdnimg.cn/20190701172329606.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaWNhbzE5OTA=,size_16,color_FFFFFF,t_70" style="zoom: 80%;" />

​	

​		常用的激活函数通常为S型(Sigmoid)激活函数,双曲正切(Tanh)激活函数、线性修正单元(ReLU)激活函数等

### 												Pytorch 常用激活函数

|     层对应的类      |      功能       |
| :-----------------: | :-------------: |
| `torch.nn.Sigmoid`  | Sigmoid激活函数 |
|   `torch.nn.Tanh`   |  Tanh激活函数   |
|   `torch.nn.ReLU`   |  ReLU激活函数   |
| `torch.nn.Softpuls` |  ReLU激活函数   |



### sigmoid激活函数

 		**sigmoid激活函数值的范围为（0， 1），经过它激活得到的数据为非0均值；sigmoid 激活函数具有双向饱和性，即在一定数据范围内，其导数趋于 0 收敛。且其导数范围为（0，0.25），且不在（−3，3）的数据导数值很小，在反向传播过程时，导数相乘很容易造成梯度弥散；sigmoid 激活函数求导过程计算量较大，模型训练的时间复杂度较高。**

`torch.nn.Sigmoid()`对应的Sigmoid激活函数,也叫logistic激活函数,计算方式为:

​	![](https://img-blog.csdnimg.cn/20190701171658198.png)

​	对应的导函数为:

​		f(x)=f(x)⋅(1−f(x))

​	**Sigmoid函数用于将模型的输出进行归一化到(0,1)区间，普遍应用于分类模型中的预测概率值，**









### **tanh 激活函数**

tanh 激活函数解决了 sigmoid 激活函数非0均值的问题 ，且其导数范围为（0，1），从而略微缓减了sigmoid 激活函数梯度弥散的问题；但 tanh 激活函数存在的双向饱和性仍然使得梯度弥散问题存在，且模型训练的时间复杂度较高

​		![Tanh Activeta Functions](https://img-blog.csdnimg.cn/0827513dc9e648b0bebc9cd877577040.png)

<img src="https://img-blog.csdnimg.cn/48cf727bbcac45119ec2ed763dc30dd9.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5aSn6b6Z5YmR56We,size_13,color_FFFFFF,t_70,g_se,x_16" style="zoom:50%;" />					

### 可视化激活函数图像

```python
import torch
import matplotlib.pyplot as plt

x = torch.linspace(-6, 6, 100)
sigmoid = torch.nn.Sigmoid() # Simgoid 激活函数
Csigmoid = sigmoid(x)
tanh = torch.nn.Tanh() #Tanh激活函数
Ctanh = tanh(x)
relu = torch.nn.ReLU() # ReLu激活函数
Crelu = relu(x)

plt.figure(figsize=(14, 3)) # 可视化数据
plt.subplot(1, 4, 1)
plt.plot(x.data.numpy(), Csigmoid.data.numpy(), "r-")
plt.title("Sigmoid")
plt.grid()
plt.subplot(1, 4, 2)
plt.plot(x.data.numpy(), Ctanh.data.numpy(), "r-")
plt.title("Tanh")
plt.grid()
plt.subplot(1, 4, 3)
plt.plot(x.data.numpy(), Crelu.data.numpy(), "r-")
plt.title("ReLu")
plt.grid()
plt.show()
```

输出：

​	![Activeta Layers](F:\PenRoad\Neural NetWork\Pytorch快速入门\Pictrue\Activeta_layers.png)

​		



参考链接:

​		[Pytorch激活函数解析](https://blog.csdn.net/weixin_38881440/article/details/115285682?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166618368116782248547962%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=166618368116782248547962&biz_id=0&utm_medium=distribute.pc_chrome_plugin_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~hot_rank-2-115285682-null-null.nonecase&utm_term=Pytorch%20%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0&spm=1018.2226.3001.4450)

​		



## 循环层



### 循环层说明:

​		循环神经网络的来源是为了刻画**一个序列当前的输出与之前信息的关系**。从网络结构上，循环神经网络会记忆之前的信息，并利用之前的信息影响后面结点的输出。即：循环神经网络的**隐藏层之间的结点是有连接的**，隐藏层的输入不仅包括输入层的输出，还包括上一时刻隐藏层的输出。其中双向循环神经网络`（Bidirectional RNN, Bi-RNN）`和长短期记忆网络`（Long Short-Term Memory networks，LSTM)`是常见的循环神经网络

​		

### **为什么要使用循环神经网络**

​		RNN背后的想法是利用顺序的信息。 在传统的神经网络中，我们假设所有输入（和输出）彼此独立。 如果你想预测句子中的下一个单词，你就要知道它前面有哪些单词，甚至要看到后面的单词才能够给出正确的答案。 RNN之所以称为循环，就是因为它们对序列的每个元素都会执行相同的任务，所有的输出都取决于先前的计算。 从另一个角度讲RNN的它是有“记忆”的，可以捕获到目前为止计算的信息。 理论上，RNN可以在任意长的序列中使用信息，但实际上它们仅限于回顾几个步骤。 循环神经网络的提出便是基于记忆模型的想法，期望网络能够记住前面出现的特征，并依据特征推断后面的结果，而且整体的网络结构不断循环，因为得名循环神经网络。

以下是RNN在NLP中的一些示例： **语言建模与生成文本**

- **机器翻译**：机器翻译类似于语言建模，我们的输入源语言中的一系列单词，通过模型的计算可以输出目标语言与之对应的内容。
- **语音识别**：给定来自声波的声学信号的输入序列，我们可以预测一系列语音片段及其概率，并把语音转化成文字
- **生成图像描述**：与卷积神经网络一起，RNN可以生成未标记图像的描述。



### Pytorch 循环类与功能

|    循环层对应的类     |           功能           |
| :-------------------: | :----------------------: |
|   `torch.nn.RNN()`    |       多层RNN单元        |
|   `torch.nn.LSTM()`   |  多层长短期记忆LSTM单元  |
|    `torch.nn.GRU`     |   多层门限循环GRU单元    |
| `torch.nn.RNNCell()`  |    一个RNN循环层单元     |
| `torch.nn.LSTMCell()` | 一个长短期记忆LSTM()单元 |
| `torch.nn.GRUCell()`  |   一个门限循环GRU单元    |

​		

### Explorer-`torch.nn.RNN()`输入一个多层的Elman RNN:

​		激活使用:`tanh` Or `ReLU`

输入到序列的元素,RNN每层的计算公式为:

​	**h~t~=tanh(W~ih~x~t~+b~ih~+W~hh~X~t-1~+b~hh~)**

#### 参数解释:

​	h~t~为时刻t的隐状态,X~t~是上一层时刻t的隐状态,或是第一层在时刻t的输入.若nonlinearity=relu,则使用ReLU函数代替tanh函数作为激活函数

#### `torch.nn.RNN()`循环层的参数、输入和输出

`input_size()`输入X的特征数量

`hidden_size()`:隐层的特征数量

`num_layers`:RNN网络的层数

`nonlinearity`指定非线性函数使用tanh或是relu,默认是tanh

`bias`:如果是Flase,那么RNN层就不会使用偏置权重,默认是True

`batch_first`如果是True,那么输入和输出的Shape应该是[batch_size,time_step,feature]

`dropout`如果值非0,那么除了最后一层外,其他RNN层的输出都会套上一个dropout层,默认为0

`bidirectional`如果是True,将会变成一个双向RNN,默认为False.

---

​		RNN的输入为Input和h_0,其中input是一个形状为(seq_len,batch,input_size)的Tensor.h_0则是一个形状为(num_layers x num_directions,batch,hidden_size)保存着初始隐状态的Tensor.如果不提供就默认为0;如果是双向RNN,num_directions等于2,否则为1

​		RNN的输出为output_和h_n,其中:

​		output是一个形状为(seq_len,batch,hidden_size * num_directions)Tensor,保存着RNN最后一层的输出特征.如果输入是被填充过的序列,那么输出也是被填充过的序列

​		h_n是一个形态为(num_layers * num_directions,batch,hidden_size)的Tensor,保持着最后一个时刻的隐状态

​		



## 全连接层

### 全连接层解释:

​		**全连接层，是每一个[结点](https://baike.baidu.com/item/结点/9794643?fromModule=lemma_inlink)都与上一层的所有结点相连，用来把前边提取到的特征综合起来。由于其全相连的特性，一般全连接层的参数也是最多的。**(By Baidu)

> ​		通常说全连接层是指一个由多个神经元所组成的层,其所有的输出和该层所有输入都有连接,即每个输入都会影响所有神经元的输出.

​		**一个简单的网络具体介绍一下推导过程**

​		

![pictrue](https://bkimg.cdn.bcebos.com/pic/e61190ef76c6a7ef13ab3ed4f1faaf51f2de66b7?x-bce-process=image/resize,m_lfit,w_398,limit_1)

​		

​		

### 引入:

​		在Pytorch中的`nn.Linear()`表示线性变换,全连接层可以看作是`nn.Linear()`表示线性变层再加上一个激活函数层所构成的结构

​		

### `nn.Linear()`全连接层操作及相关参数	

##### 		函数定义:`torch.nn.Linear(in_features,out_features,bias=True)`



##### 参数说明:

​		`in_features`:每个输入样本的特征数量.

​		`out_features`:每个输出样本的特征数量.

​		`bias:若设置为False`,则该层不会学习偏置.默认为False

​		`torch.nn.Linear()`输入为(N,in_features)的Tensor,输出为(N,out_features)的Tensor



### 总结:

​		全连接层的应用范围非常广泛,`只有全连接层组成的网络是全连接神经网络`,可用于**数据分类**或**回归预测**,卷积神经网络和循环神经网络的末端,通常会由多个全连接层组成.





## Pytorch中数据操作和预处理

### 		解释：

​	  在Pytorch中torch.utils.data模块包含一些常用的数据预处理的操作,主要用于数据的读取、切分、准备等.



### 												常用的数据操作类

|            数据操作类            |                    功能                    |
| :------------------------------: | :----------------------------------------: |
| `torch.utils.data.TensorDataset` |             将数据处理为Tensor             |
| `torch.utils.data.ConcatDataset` |               连接多个数据集               |
|    `torch.utils.data.Subest`     |          根据索引获取数据集的子集          |
|  `torch.utils.data.DataLoader`   |                 数据加载器                 |
| `torch.utils.data.random_split`  | 随机将数据集拆分为给定长度的非重叠新数据集 |



### 总结

​	使用这些类能够对高维数组、图像等各种类型的数据进行预处理,以便深度学习模型使用.针对文本数据的处理可以使用`torchtext`库进行相关的数据准备操作.

---

==下面针对分类和回归模型模型,在高维数组、图像及文本数据上的相关预处理和数据准备工作进行介绍.==

---

​	

## 高维数组

### 前言：

​	 ==在很多情况下,我们需要从文本(如csv文件)中读取高维度数组数据,这类数据的特征是每个样本都有很多个预测变量(特征)和一个被预测变量(目标标签)，特征通常是数值变量或者离散变量,被预测变量如果是连续的值,则对应回归问题的预测;如果是离散变量,则对应着分类问题.在使用Pytorch建立模型对数据进行学习时,通常要对数据进行预处理,并将它们转换为网络需要的数据形式.==



### Sample:

1. 回归数据

   针对全连接神经网络模型回归问题的数据准备,首先加载对应的模块,然后读取数据,Code：、

   ```python
   import torch
   import torch.utils.data as Data
   from sklearn.datasets import load_boston, load_iris
   
   boston_X, boston_Y = load_boston(return_X_y=True)
   print("Boston_X.dytpe:", boston_X.dtype)
   print("Boston_Y.dtype:", boston_Y.dtype)
   ```

   输出：`Boston_X.dytpe: float64
   	  Boston_Y.dtype: float64`

输出的数据集的特征和被预测变量都是Numpy的64位浮点数据.而使用Pytorch时需要转换为torch32位浮点型的Tensor,故需要将训练集boston_X和boston_Y转化为32位浮点型Tensor:

```python
train_xt = torch.from_numpy(boston_X.astype(np.float32))
train_yt = torch.from_numpy(boston_Y.astype(np.float32))
print("train_xt.dtype is :", train_yt.dtype)
print("train_yt.dtype is :", train_yt.dtype)
```

输出:`train_xt.dtype is : torch.float32
     train_yt.dtype is : torch.float32`

先将Numpy数据转化为32位浮点型,然后使用`torch.from_numpy()`函数,将数组转化为Tensor.在训练全连接神经网络时,通常一次使用一个batch的数据进行权重更新,`torch.utils.data.DataLoader`函数可以将输入的数据集(包括数据特征Tensor和被预测变量Tensor)获取一个加载器,每次迭代可使用一个batch的数据,Code:

```python
# 将Numpy转化好的torch.float32类型的数据类型再转化为Tensor且使用TorchDataset将X和Y整理到一起
train_data = Data.TensorDataset(train_xt, train_yt)
# 定义一个数据加载器,将训练数据进行批量处理
train_loader = Data.DataLoader(
    dataset=train_data,  # 使用的训练集
    batch_size=64,  # 批处理样本大小
    shuffle=True,  # 每次跌代前打乱数据
    num_workers=0,  # 该例子中即把下面这行注释掉、删掉或者num_workers设为0。该语句表示调用 1+num_workers 个进程。Linux系统，该参数正常，不需担心。
)

# 检查训练集的一个batch的样本维度是否正确
for step, (b_x, b_y) in enumerate(train_loader):
    if step > 0:
        break

# 输出训练数据的尺寸和标签尺寸及数据类型
print("b_x.shape is :", b_x.shape)
print("b_y.shape is :", b_y.shape)
print("b_x.btype is :", b_x.dtype)
print("b_y.btype is :", b_y.dtype)
```

输出:`b_x.shape is : torch.Size([64, 13])
	 b_y.shape is : torch.Size([64])
     b_x.btype is : torch.float32
     b_y.btype is : torch.float32`



在上面数据中,首先使用Data.TensorDataset()将训练数据X,Y放在一起组成train_data,然后使用Data.DataLoader()定义一个数据加载器,每64个样本为一个batch,最好使用for循环获得一次加载器的输出内容b_x和b_y,均为torch.float32数据类型



2.分类数据

​	分类数据和回归数据不同点在于,分类数据的被预测变量为离散类别变量,所以在使用Pytorch定义的网络模型中时,默认的预测标签是64位有符号整型数据.

```
# 数据分类准备
iris_x, iris_y = load_iris(return_X_y=True)
print("iris_x.dtype :", iris_x.dtype)
print("iris_y.dtype :", iris_y.dtype)
```

输出:`iris_x.dtype : float64
	 iris_y.dtype : int32`

​	上述代码执行读取数据,然后查看数据的特征和标签的数据类型.在上述输出中可得知,该数据集的特征数据(X)为64浮点型,标签(Y)为64为整型.在torch构建的网络中,X默认数据格式是torch.float32,所以转化为Tensor时,数据的特征要转化为32位浮点型,数据的类别标签要转化为64位有符号整数.

​	将X\Y转化为Tensor

```python
# 处理分类数据
train_xt = torch.from_numpy(iris_x.astype(np.float32))
train_yt = torch.from_numpy(iris_y.astype(np.int64))
print("train_xt.dtype is :", train_xt.dtype)
print("train_yt.dtype is :", train_yt.dtype)
```

输出:`train_xt.dtype is : torch.float32
	 train_yt.dtype is : torch.int64`



​	准备好数据类型后,再使用`Data.TensorDataset()`和`Data.DataLoader()`定义数据加载器,Code:	

```python
# 将训练集转化为Tensor后，再使用Data.TensorDataset()和Data.DataLoader()定义数据加载器
train_data = Data.TensorDataset(train_xt, train_xt)
train_loader = Data.DataLoader(
    dataset=train_data,  # 使用的训练集
    batch_size=10,  # 批处理样本大小
    shuffle=True,  # 每次迭代前打乱数据
    num_workers=0  # Remember : 在Windows中,此参数最填0,否则将抛出ERROR
)

for step, (b_x, b_y) in enumerate(train_loader):
    if step > 0 :
        break

print("b_x.shape is:", b_x.shape)
print("b_y.shape is:", b_y.shape)
print("b_x.dtype is:", b_x.dtype)
print("b_y.dtype is:", b_y.dtype)
```

输出:`b_x.shape is: torch.Size([10, 4])
	 b_y.shape is: torch.Size([10, 4])
     b_x.dtype is: torch.float32
	 b_y.dtype is: torch.float32`



​	上述batch使用了10个数据样本,并且数据的类型已经正确转化.



## 图像数据

### 	`torchvision`中的datasets模块包含多重常用的分类数据集下载及导入函数,可以方便导入数据及验证建立的模型效果.datasets模块所提供的部分常用图像数据.

|     图像数据集对应的类     |                  描述                  |
| :------------------------: | :------------------------------------: |
|     `datasets.MNIST()`     |             手写字体数据集             |
| `datasets.FashionMNIST()`  |       衣服、鞋子、包等10类数据集       |
|    `datasets.KMNIST()`     |           一些文字的灰度数据           |
| `datasets.CocoCaptions()`  |       用于图像标注的MS COCO数据        |
| `datasets.CocoDetection()` |         用于检测的 MS COCO数据         |
|     `datasets.LSUN()`      |     10个场景和20个目标的分类数据集     |
|    `datasets.CIFAR10()`    |            CIFAR10类数据集             |
|   `datasets.CIFAR1OO()`    |            CIFAR100类数据集            |
|     `datasets.STL10()`     | 包含10类的分类数据集和大量的未标记数据 |
|  `datasets.ImageFolder()`  |   定义一个数据加载器从文件夹读取数据   |



​	`torchvision`中的transforms模块可以针对每张图像进行预处理操作,在该模块中提供以下常用图像操作.

​	

|           数据集对应的类           |                             描述                             |
| :--------------------------------: | :----------------------------------------------------------: |
|       `transforms.Compose()`       |                 将多个transform组合起来使用                  |
|        `transforms.Scale()`        |                按照指定的图像尺寸对图像进调整                |
|      `transform.CenterCrop()`      |               将图像进行中心切割,得到给定大小                |
|      `transform.RandomCrop()`      |                   切割中心点的位置随机选取                   |
| `transform.RandomHorizontaIFlip()` |                       图像随机水平翻转                       |
|   `transform.RandomSizedCrop()`    |          将给定的图像随机切割,然后变化为指定的大小           |
|         `transform.Pad()`          |              将图像所有边用给定的Pad value填充               |
|       `transform.ToTensor()`       | 把一个取值范围是[0,255]的PIL图像或形状为[H,W,C]的数组,转化成形状为[C,H,W],取值是[0,1,0]的张量(torch.FloadTensor) |
|      `transforms.Normalize()`      |                   将给定图像进行规范化操作                   |
|     `transform.Lambda(lambd)`      |           使用lambd作为转换器,可自定义图像操作方式           |
|                                    |                                                              |
|                                    |                                                              |



### Use:

​	预处理:

​	从`torchvision`中的datasets模块中导入数据并预处理

```python
import torch
import torch.utils.data as Data
from torchvision.datasets import FashionMNIST
```

​	从`文件夹`中导入数据并进行预处理

```python
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
```

​	

### Sample :

#### 从`torvision`中的datasets模块导入数据并预处理数据

导入FashionMNIST数据集为例,该数据集包含一个60000张28 * 28的灰度图片作为训练集,以及10000张28 * 28的灰度图片作为测试集.数据共10类,分别是鞋子、T恤、连衣裙...饰品类图像

Code:

```python
import torch.utils.data as Data
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST

# 使用FashionMNIST训练集,准备训练数据集
train_data = FashionMNIST(
    root="./data/FashionMNIST",  #训练集的路径
    train=True, #只使用训练集
    transform=transforms.ToTensor(),
    download=True  # 如果没有训练集设置为True反之False
)

# 定义一个数据加载器
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

print("train_loader 的 batch 数量为:", len(train_loader))
```

输出:`train_loader 的 batch 数量为: 938`



上述Function解释:

​		通过`FashionMNIST`函数来导入数据,该函数中root参数用于指定需要导入的训练集所在的路径(如果指定路径下已有该训练集,需要把参数`download`设置为`False`,如果路径下没有训练集,则需要设置`download`为`True`).参数`train`的取值为True或False表示,表示导入的数据是训练集(60000张图片)或测试集(10000张图片).参数`transform`用于指定数据集的变换,`transform=transforms.ToTensor()`表示将数据中的像素值转换到0-1之前,并且将图像数据从形状为[H,W,C]转换形状为[C,H,W]

​		在数据导入后需要利用数据加载器`DataLoader()`将整个数据集切分为多个**batch**,用于网络优化时利用梯度下降算法进行求解.在函数中`datasets`参数用于指定使用的训练集;`batch_size`参数指定每个batch使用的样本数量;`shuffle=True`表示从数据集中获取每个批量图片前打乱数据;`num_workers`参数用于指定导入数据使用的进程数量(和并行处理相似).经过处理后该训练集包含938个batch

​		对训练数据集进行处理后,可以使用相同的方法对测试集进行处理,也可以使用如下对测试集进行处理.

```python
test_data = FashionMNIST(
    root="./data/FashionMNIST",
    train=False,
    download=False
)

#  为数据添加一个通道维度,并且取值范围缩放到0~1之间
test_data_x = test_data.data.type(torch.FloatTensor) / 255.0
test_data_x = torch.unsqueeze(test_data_x, dim=1)
test_data_y = test_data.targets  # 测试集标签
print("test_data_x.shape is :", test_data_x.shape)
print("test_data_y.shape is :", test_data_y.shape)
```

输出：`test_data_x.shape is : torch.Size([10000, 1, 28, 28])
	  test_data_y.shape is : torch.Size([10000])`

​		使用`FashionMNIST()`函数导入数据,使用`train=False`参数指定导入测试集,并将数据集的像素值除以255.0,使像素值转化到0~1之间,再使用函数`torch.unsqueeze()`为数据添加一个通道,即可得到测试数据集.在test_data中使用`test_data.data`获取图像数据,使用`test_data.target`获取每个图像所对应的标签.



#### 从文件夹中导入数据并进行预处理:

在`torchvision`的`datasets`模块中包含有`ImageFolder`函数,该函数可以读取如下格式的数据集:

root/dog/xxx.png

root/dog/xxx.png

···

root/cat/xxx.png



在读取文件夹中的图像前,需要对训练数据集进行变换操作(预处理):

```python
train_data_transforms = transforms.Compose(
    transforms.RandomResizedCrop(224),  # 随机长宽裁剪比为: 224 * 224
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # 转化为Tensor并归一化至[0 ,1]
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #图像标准化处理
)
```

​		上述代码,使用`transforms.Compose()`函数可以将多个变化操作组合在一起,其中train_data_transforms包含了将图像随机剪切为224 * 224,依概率p=0.5水平翻转、转化为Tensor并归一化至0~1、图像标准化处理等操作.



现在使用`ImageFolder`函数读取图像,其中的`transforms`参数指定读取图像时对每张图像所做的变换,图像的取值范围为Tensor(-2.1179)~Tensor(2.6400).

读取图像后,同样使用了`DataLoader()`函数创建了一个数据加载器.从输出结果可以发现,共读取了1张图像,每张图像是224 * 224的RGB图像,经过转化后,图像的像素值在-2.1008~2.4134之间.

Code:

```python
import torch.utils.data as Data
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST, ImageFolder

train_data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),  # 随机长宽裁剪比为: 224 * 224
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # 转化为Tensor并归一化至[0 ,1]
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 图像标准化处理
])

train_data_dir = "data/Image/"  # 设置图像路径
train_data = ImageFolder(train_data_dir, transform=train_data_transforms)

train_data_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

print("数据集的Label:", train_data.targets)

# 获取一个batch数据
for step, (b_x, b_y) in enumerate(train_data_loader):
    if step > 0:
        break

print("b_x Shape is", b_x.shape)
print("b_y Shape is", b_y.shape)
print("图像的取值范围为:", b_x.min(), "~", b_x.max())
```

输出:`数据集的Label: [0]
     b_x Shape is torch.Size([1, 3, 224, 224])
     b_y Shape is torch.Size([1])
     图像的取值范围为: tensor(-2.1008) ~ tensor(2.4134)`



## 文本数据

###		导言:

对文本数据进行分类是深度学习任务中常见的应用,但是Pytorch建立的深度学习网络不能直接作用于文本数据,需要对文本数据进行相应的预处理.

在指定文件夹中,包含两个文本数据的数据集train.csv和test.csv，在每个文件中均包含两列数据,分别表示文本对应的标签变量label和表示文本的内容变量test.

​		

![](https://pic3.zhimg.com/80/v2-69383f73f912cf2f04e7b893d914523a_720w.webp)

## Warning:受版本影响,书中部分函数在新版本中被删除,顾等待二次回顾补充.
