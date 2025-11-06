# **Experiment4--MLP_MNIST** (实验4--实现简单多层感知机)
##### This experiment constructs a simple Multi-Layer Perceptron(MLP) based on the MNIST dataset to realize handwritten digit classification and performance evaluation. 
###### 本实验基于 MNIST 手写数字数据集，构建简单多层感知机（MLP），实现手写数字的分类任务与性能评估。

## 1.Exprimental Purpose
##### Based on the handwritten digit dataset,build a simple Multi-Layer Perceptron(MLP),master the neural network training process,realize handwritten digit classification and verify model performance.
###### 基于 MNIST 手写数字数据集，搭建简单多层感知机（MLP），掌握神经网络训练流程，实现手写数字分类并验证模型性能。

##

## 2.Exprimental Content
##### Construct an MLP model with input,hidden and output layers,use the SGD optimizer and cross-entropy loss function,and evaluate loss and accuracy on the test set after 10 training epochs.
###### 构建含输入层、隐藏层和输出层的 MLP 模型，使用 SGD 优化器和交叉熵损失函数，经 10 轮训练后在测试集评估损失与准确率。
##
#### 2.1.MNIST Dataset Loading and Processing
###### Import Pytorch and related tool libraries
###### 导入 PyTorch 及相关工具库
```
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets,transforms
```
##
###### Set the batch size and CPU computing device,define data loaders for the training and test sets,automatically download/read the MNIST dataset and convert its format via ToTensor(),configure batch size and shuffling parameters.
###### 设置批次大小与 CPU 计算设备，定义训练集和测试集的数据加载器，自动下载 / 读取 MNIST 数据集并通过 ToTensor () 转换格式，同时配置批次大小与打乱等参数。
```
batch_size =2048
device = torch.device('cpu')
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data',train=True,download=True,transform=transforms.Compose([transforms.ToTensor(),])),
    batch_size=batch_size,shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data',train=False,transform=transforms.Compose([transforms.ToTensor(),])),
    batch_size=batch_size,shuffle=True
)
```
> 代码中的一些参数解释：

> 'data'：数据集存储路径：将 MNIST 数据下载/读取到当前目录的 'data' 文件夹 

> train=True：指定为训练集：MNIST 分为训练集（60000 张图）和测试集（10000 张图），True 表示加载训练集

> transforms.Compose([...]) 是 PyTorch 的「预处理流水线」，用于按顺序执行多个数据转换操作，这里只包含一个操作 transforms.ToTensor()
##
###### Finally check the original data dimensions through dataset attributes.
###### 最终可通过数据集属性查看原始数据维度。
```
train_loader.dataset.data.shape,
```
###### result : (torch.Size([60000, 28, 28]),)
##
#### 2.2.
