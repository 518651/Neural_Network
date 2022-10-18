# `Torch.nn`构建网络#2

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

