# Networks
Train the Mobilenet in flowers dataset
Mobilenet V1 75.3%
Mobilenet V2 79.4%
Mobilenet V3 82.1%


![QQ截图20210720194130](https://user-images.githubusercontent.com/52600391/126318067-66b47327-8f4e-4f17-8b7d-2afe54ee4708.png)

# Batch Normalization
  目的：对于Conv2的feature map无法满足某一分布规律，提出BN使得feature map满足均值为0，方差为1的分布规律。
  
  处理：对于一个d维的输入x，我们将对它的每一个维度进行标准化处理
  
  参数：均值和方差 是由特征数据计算而来;\gama 和 \beita 是由训练学习而来.
  
  注意：1. 训练时，将training设为 true，推理时training设为 false；
        2. batch size 尽量设大
        3. 将 BN 放在卷积和激活层之间，卷积层不要使用 bias

# AlexNet
  
  1. 首次使用GPU进行网络加速训练
  2. 使用ReLU激活函数，而不是Sigmoid
  3. 使用LRN局部响应归一化
  4. 在全连接前使用Dropout,减少过拟合

# VGG
  
  1. 在保证相同感受野的情况下，使用小的卷积核替换大的卷积核

# GoogleNet

  1. 引入Inception结构（融合不同尺度的特征信息）
  2. 使用1x1的卷积核进行降维以及隐射处理
  3. 添加两个辅助分类器帮助训练
  4. 丢弃全连接层，使用平均池化层
  
# ResNet

  1. 超深的网络结构
  2. 提出Residual模块
  3. 使用BN加速训练（丢弃dropput）

  随着网络加深，网络效果会变差，主要原因是：
  1. 梯度消失或梯度爆炸（BN和初始化）
  2. 退化问题（残差结构）
