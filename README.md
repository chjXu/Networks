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
  
  注意：
  
  1. 训练时，将training设为 true，推理时training设为 false；
  2. batch size 尽量设大
  3. 将 BN 放在卷积和激活层之间，卷积层不要使用 bias

  优势：
  1. 加快网络训练
  2. 避免了人工调参

# AlexNet
  
  1. 首次使用GPU进行网络加速训练
  2. 使用ReLU激活函数，而不是Sigmoid，解决梯度消失问题，收敛更快
  3. 使用LRN局部响应归一化（对于输出的N通道特征像素值进行归一化处理），提高准确率
  4. 在全连接前使用Dropout,减少过拟合

# VGG
  
  1. 在保证相同感受野的情况下，使用小的卷积核替换大的卷积核
  2. 2个3x3的卷积核替换1个5x5的卷积核

  优点：
  1. 网络结构简单，采取相同的卷积核尺寸核最大池化尺寸
  2. 验证了通过加深网络可以取得更好的效果

  缺点：
  1. 耗费更多的计算资源（主要是全连接层的问题）

# GoogleNet

  1. 引入Inception结构（融合不同尺度的特征信息）
  2. 使用1x1的卷积核进行降维以及隐射处理
  3. 添加两个辅助分类器帮助训练
  4. 丢弃全连接层，使用平均池化层

# MobileNet V1, V2, V3
  
  ## V1
  1. 深度可分卷积（Depthwise Convolution）。本质上是将深度channel和特征的大小分开处理。
  2. 超参数a，b。a 卷积核个数倍率，a 越小，网络参数量越小，但准确率降低；b 为输入图像分辨率参数。

传统卷积 VS DW卷积

    传统卷积核channel = 输入特征矩阵channel
    输出特征矩阵channel = 卷积核个数
    
    DW卷积核channnel = 1
    输入特征矩阵channel = 卷积核个数 = 输出特征矩阵channel
因此，DW卷积与传统卷积最大的不同就是：传统卷积的卷积核channel为输入特征矩阵的channel。而DW卷积的卷积核channel为1，个数为输入特征矩阵通道数。

  ## V2
  1. Inverted Residuals(倒残差结构)。 倒残差结构先1x1升维，3x3 DW卷积， 在1x1降维。
  2. Linear Bottlenecks。在Bottlenecks的最后一层的激活函数使用线性激活函数，目的是为了防止ReLU对低维信息不敏感。

  ## V3
  1. 更新block（加入注意力机制，更新激活函数）
  2. 使用NAS搜索参数
  3. 重新设计耗时层结构
  
# ResNet

  1. 超深的网络结构
  2. 提出Residual模块
  3. 使用BN加速训练（丢弃dropput）

  随着网络加深，网络效果会变差，主要原因是：
  1. 梯度消失或梯度爆炸（BN和初始化）
  2. 退化问题（残差结构）


# Transformer
  
  1. transformer之前主要是应用于NLP中
  2. transformer的提出主要是解决RNN，LSTM中计算无法并行化的问题（记忆化无限长）
  3. self-attention

      ![image](https://user-images.githubusercontent.com/52600391/130730453-d50ce5e8-df8d-4906-b0fd-1a1d55480ff4.png)
      
  4. Multi-head self-attention
     Multihead = concat(head1, head2,...,headn)W
     先将输入与Q，K，W权重分别相乘之后，输出q,k,v特征，然后根据head个数将q,k,v均分为(q11,q12, k11,k12, v11,v12和q21,q22, k21,k22, v21,v22),再将（q11, k11,v11, q21, k21, v21）组成head1，head2类似。
     
# Vision Transformer(ViT)
  纯Transformer模型
  
  ![image](https://user-images.githubusercontent.com/52600391/130731719-3d8f64f3-0491-419c-9e76-f6d00ba65dfd.png)
  
  1. 图片分割为Patch
  2. 输入Embedding层。对于标准的Transformer模块，要求输入的是token（向量）序列，即二维矩阵[num_token, token_dim]。 在代码中是通过卷积实现的。在输入到Transformer Encoder之前需要加上class token。
    position embedding（可训练参数）使用余弦相似度。
  4. MLP Head分类
    Linear层
    
    
  Hybrid(CNN with Transformer) 混合模型较纯Transformer在训练次数变大时，正确率是不如纯模型的
  
    ResNet50提取特征：
      采用StdConv2d而不是Conv2d;
      将BN替换为GroupNorm;
      将stage4中的3个block移至stage3中。
  
 ![image](https://user-images.githubusercontent.com/52600391/130733424-dc0d6325-3bc1-41c6-99cf-feff7be6f3ba.png)



