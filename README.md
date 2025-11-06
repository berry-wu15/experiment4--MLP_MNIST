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
###### 
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
##
###### 
