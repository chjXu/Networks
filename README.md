# Networks
Train the Mobilenet in flowers dataset
Mobilenet V1 75.3%
Mobilenet V2 79.4%
Mobilenet V3 82.1%


![QQ截图20210720194130](https://user-images.githubusercontent.com/52600391/126318067-66b47327-8f4e-4f17-8b7d-2afe54ee4708.png)

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
