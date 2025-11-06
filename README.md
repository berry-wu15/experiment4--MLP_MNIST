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
##### Import Pytorch and related tool libraries
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
##### Set the batch size and CPU computing device,define data loaders for the training and test sets,automatically download/read the MNIST dataset and convert its format via ToTensor(),configure batch size and shuffling parameters.
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
###### 代码中的一些参数解释：
###### 'data'：数据集存储路径：将 MNIST 数据下载/读取到当前目录的 'data' 文件夹 
###### train=True：指定为训练集：MNIST 分为训练集（60000 张图）和测试集（10000 张图），True 表示加载训练集
###### transforms.Compose([...]) 是 PyTorch 的‘预处理流水线’，用于按顺序执行多个数据转换操作，这里只包含一个操作 transforms.ToTensor()
###### transforms.ToTensor()：将 MNIST 原始的「图片格式」转换为 PyTorch 能处理的「张量（Tensor）格式」

##
##### Finally check the original data dimensions through dataset attributes.
###### 最终可通过数据集属性查看原始数据维度。
```
train_loader.dataset.data.shape,
```
###### result: 
###### (torch.Size([60000, 28, 28]),)
##
###### 代码中的一些参数解释：
###### train_loader.dataset：DataLoader 类有一个内置属性 dataset，专门用于返回它所绑定的「原始数据集实例」。
###### dataset.data:datasets.MNIST 类（PyTorch 内置的数据集类）有一个专属属性 data，用于存储「原始图片数据」
###### .shape:张量（Tensor）的内置属性，用于返回张量的「维度大小」，格式为 torch.Size([维度1, 维度2, ...])。
###### 对 MNIST 训练集来说，shape 就是 torch.Size([60000, 28, 28])，对应：
###### 60000：训练集样本总数（MNIST 训练集固定 60000 张图）
###### 28：每张图片的高度（像素）
###### 28：每张图片的宽度（像素）

##
#### 2.2.MLP Model Definition and Training Component Initialization
##### Define a simple Multi-layer Perceptron(MLP) based on fully connected layers,including the network structure from input layer to hidden layer(784-->128) and hidden layer to output layer(128-->10), with ReLU activation function introducing non-lnearity.
###### 定义基于全连接层的简单多层感知机（MLP），包含输入层到隐藏层（784→128）、隐藏层到输出层（128→10）的网络结构，通过 ReLU 激活函数引入非线性。
```
#简单的多层感知机（MLP，也叫人工神经网络）
#MLP 是「全连接神经网络」：层与层之间的每个神经元都和下一层所有神经元相连，是深度学习最基础的网络结构。
class mlp(nn.Module):
#nn.Module: PyTorch 中所有神经网络模型的「基类（父类）」，继承它才能使用 PyTorch 提供的模型训练、参数优化、GPU 迁移等核心功能（比如 to(device)、parameters() 等）。
    def __init__(self):
        super(mlp,self).__init__()  # 调用父类 nn.Module 的构造函数
        self.l1 = nn.Linear(784,128)  
        #28*28=784  ,PyTorch 内置的「全连接层」（也叫线性层）
        #in_features=784输入特征数 ——MNIST 图片是 28×28，展平后是 28×28=784 个像素（每个像素是一个特征）。
        self.l2 = nn.Linear(128,10)

    def forward(self,x):
        a1 = self.l1(x)
        #调用 self.l1(x) 时，nn.Linear 的 forward 会执行：a1 = x @ W + b（W 是 784×128 的权重矩阵，b 是 128 维的偏置）；
        #输出 a1 的形状是 (batch_size, 128)，正好作为下一层的输入。
        x1 = F.relu(a1)
        a2 = self.l2(x1)
        x2 = a2
        return x2
```
###### 代码中的一些参数解释：
###### 输入（展平的 MNIST 图片）→ 全连接层（784→128）→ ReLU 激活 → 全连接层（128→10）→ 输出（10 类预测分数）
###### 用形状表示：(batch_size, 784) → (batch_size, 128) → (batch_size, 128) → (batch_size, 10)
##
##### 
```
model = mlp().to(device)   #创建模型实例并指定计算设备
optimizer = optim.SGD(model.parameters(),lr=0.1)
model
```
###### result : 
###### mlp( (l1): Linear(in_features=784, out_features=128, bias=True)  
###### (l2): Linear(in_features=128, out_features=10, bias=True) )
##
###### 代码中的一些参数解释：
###### SGD 是「随机梯度下降（Stochastic Gradient Descent）」的缩写
###### nn.Module 类的内置方法，返回模型中「所有可训练参数的迭代器」—— 这里就是 self.l1 的权重（784×128）、偏置（128 维），以及 self.l2 的权重（128×10）、偏置（10 维）。
###### 核心作用：将模型参数「绑定」到优化器，让优化器知道要更新哪些参数。如果不绑定，优化器无法找到需要调整的变量，训练会无效。
##
#### 2.3.MLP Model Trainning and Testing Process
##### Set 10 training epochs.In the training phase,load and flatten data in batches,optimize the model through gradient clearing,forward propagation,loss calculation,backpropagation,and parameter update. 
###### 设定 10 轮训练迭代，训练阶段按批次加载并展平数据，经梯度清零、前向传播、损失计算、反向传播及参数更新完成模型优化。
##### In the testing phase,switch to evaluation mode and disable gradient computation to save resources.Obtain the average test loss and accuracy by accumulating losses and counting correct predictions,and output the model performance after each training epoch.
###### 测试阶段切换至评估模式，禁用梯度计算以节省资源，通过累加损失、统计正确预测数得到平均测试损失与准确率，输出每轮训练后的模型性能。
```
epochs = 10
for epoch in range(epochs):
    model.train() ## 模型设为训练模式
    for batch_idx,(x,y) in enumerate(train_loader): # 批量遍历测试集
        x,y = x.view(x.shape[0],-1).to(device),y.to(device)
        #x 是图片张量（shape: (batch_size, 1, 28, 28)），y 是标签（shape: (batch_size,)，0-9 的数字）
        #数据展平：x.shape[0] 是批量大小（batch_size），-1 表示「自动计算剩余维度」—— 把 (batch_size, 1, 28, 28) 的图片张量，
        #展平为 (batch_size, 784)（匹配 MLP 输入要求：784 个像素特征）。
        optimizer.zero_grad()
        #「梯度清零」—— 每次批次训练前，必须清空上一轮的梯度（PyTorch 会默认累积梯度），否则梯度会叠加，导致参数更新混乱。
        output = model(x)
        #前向传播 —— 调用模型的 forward 方法，输入展平后的图片 x，输出 10 个类别的预测分数（shape: (batch_size, 10)）。
        loss = F.cross_entropy(output,y)
        loss.backward()
        optimizer.step()  #参数更新

    model.eval()  #模型切换到评估模式
    correct = 0
    test_loss = 0
    with torch.no_grad(): #禁用梯度计算 —— 测试阶段不需要更新参数
        for batch_idx,(x,y) in enumerate(test_loader):
            x,y = x.view(x.shape[0],-1).to(device),y.to(device)
            output = model(x)
            test_loss +=F.cross_entropy(output,y) 
            #累加每批次的测试损失
            #F 是 torch.nn.functional 的缩写（需提前导入：import torch.nn.functional as F）。它是 PyTorch 中函数式接口的集合，包含各种神经网络操作的函数实现
            pred = output.max(1,keepdim=True)[1]  #output.max(1)：在「第 1 维度（类别维度，shape=10）」上取最大值 —— 返回两个结果：(最大值, 最大值索引)。
            correct +=pred.eq(y.view_as(pred)).sum().item()
            #y.view_as(pred)：把真实标签 y 的形状调整为和 pred 一致
            #pred.eq(...)：逐元素比较预测值和真实标签，相等返回 True（1），不等返回 False（0），得到一个布尔张量。
            #.item()：把张量的数值提取为 Python 标量（方便累加）
        
        test_loss = test_loss/(batch_idx+1)  #计算平均测试损失
        acc =correct/len(test_loader.dataset) #计算测试准确率 
        print('epoch:{},loss:{:.4f},acc:{:.4f}'.format(epoch,test_loss,acc))
```
##
## 3.Experimental Reaults and Analysis
##### After training,the model achieves an accuracy of over 90% and a loss of around 0.34 on the MNIST test set,proving that a simple MLP can effectively learn handwritten digit features with stable classification performance.
###### 训练后模型在 MNIST 测试集上准确率达 90% 以上，损失降至 0.34 左右，证明简单 MLP 能有效学习手写数字特征，分类性能稳定。
##### The final testing results(loss and accuracy) are as follows :
###### epoch:0,loss:1.4814,acc:0.7388
###### epoch:1,loss:0.8339,acc:0.8310
###### epoch:2,loss:0.6112,acc:0.8624
###### epoch:3,loss:0.5090,acc:0.8769
###### epoch:4,loss:0.4524,acc:0.8857
###### epoch:5,loss:0.4144,acc:0.8925
###### epoch:6,loss:0.3894,acc:0.8962
###### epoch:7,loss:0.3721,acc:0.8983
###### epoch:8,loss:0.3555,acc:0.9030
###### epoch:9,loss:0.3443,acc:0.9052
##
## 4.Experimental Summary
#### (1)Model Learning:Mastered the core structure and working principle of Multi-Layer Perceotron(MLP),understood the functions of fully connected layers and ReLU activation function,and verified the effectiveness of a simple MLP in image classification tasks.
###### 模型上：掌握了多层感知机（MLP）的核心结构与工作原理，理解全连接层的接受数据的格式和工作方法，验证了简单 MLP 在图像分类任务中的有效性。
#### (2)Code Learning:Proficiently used PyTorch to implement data loading,model defintion,loss calculation and optimizer configuration,and mastered core progamming skills such as tensor operations,device migration,and gradient control.
###### 代码上：运用 PyTorch 实现数据加载、模型定义、损失计算与优化器配置，掌握张量操作、梯度控制等核心编程技巧。并自己动手搭建了两层简单的全连接层，定义了前向传播方法。
#### (3)Overall Process Learning:Clarified the complete workflow of deep learning classification tasks,forming a standardized experimental thinking from dataset preprocessing,model construction,training optimization to performance evaluation. 
###### 总体流程上：理清了深度学习分类任务的完整流程，从数据集预处理、模型搭建、训练优化到性能评估。
