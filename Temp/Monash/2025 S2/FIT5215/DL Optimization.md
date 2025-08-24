---
date: 2025-08-24
author:
  - Siyuan Liu
tags:
  - FIT5215
aliases:
  - summary
---
# Gradient vanishing
**定义:**
Gradients get smaller and smaller as the algorithm progresses down to the lower layers

**多发场景:**
- activate function 使用的是`Sigmoid`,`tanh`

**解决办法:**
- 避免使用`Sigmoid`,`tanh`
- 选择合适的W,b初始化
    - `Xavier`/`Glorot` 初始化（适合 `tanh`/线性模型）
    - He 初始化（适合`ReLU`)
- 通过保持中间特征的分布稳定，避免梯度消失/爆炸
    - Batch Normalization (BN) —— 训练加速、梯度稳定
    - Layer Normalization (LN) —— 特别适合 RNN/Transformer
    - Group Normalization (GN)、Instance Norm 等
- 残差连接（Residual Connections）
    - `ResNet` 中的 shortcut/skip connection 可以让梯度绕过深层传播，极大缓解梯度消失/爆炸
---
# Gradient exploding
**定义:**
The gradients can grow bigger and bigger, so many layers get insanely large weight updates, and the training diverges

**多发场景:**
- 深层神经网络
- RNN/LSTM/Bidirectional RNN

**解决办法:**
- 梯度裁剪 (Gradient Clipping)
    - 在反向传播得到梯度后，若梯度范数超过设定阈值，就按比例缩放到合理范围
    - 常用于RNNs,NLP, 但是在CNNs中不常用
- 合理的学习率
    - 可以使用 **学习率调度器 (scheduler)** 或 **自适应优化器** (如 Adam, RMSprop)
- 通过保持中间特征的分布稳定，避免梯度消失/爆炸
    - Batch Normalization (BN) —— 训练加速、梯度稳定
    - Layer Normalization (LN) —— 特别适合 RNN/Transformer
    - Group Normalization (GN)、Instance Norm 等
- 残差连接（Residual Connections）
    - `ResNet` 中的 shortcut/skip connection 可以让梯度绕过深层传播，极大缓解梯度消失/爆炸
---
# weight initialization
## What is a good weight/filter initialization?
- Break the ‘symmetry’ of the network: two hidden nodes with the same input should have different weights
    - Large initial weights has better symmetry breaking effect, help avoiding losing signals and redundant units, but could result in exploding values during back-ward and forward passes, especially in Recurrent Neural Networks
- the gradient will not vanishing or exploding
## Xavier initialization
**作用:**
Try to ensure the variance of the outputs of each layer equal to the
variance of its input. This way, signals and gradients don't shrink or amplify layer by layer in the network

**计算步骤:**
假设某一层有：
- 输入单元数：ninn_{in}nin​
- 输出单元数：noutn_{out}nout​
权重矩阵 WWW 的元素希望满足：
$$Var(Wx)≈Var(x)Var(W x) \approx Var(x)Var(Wx)≈Var(x)$$

Xavier 初始化给出了一个简单公式：
- 对**均匀分布**：
$$W \sim U\big[
-\sqrt{\frac{6}{n_in + n_out}},\sqrt{\frac{6}{n_in + n_out}}
\big]$$

- 对**正态分布**：
$$W∼N(0,2nin+nout)W \sim N\Big(0, \frac{2}{n_{in} + n_{out}}\Big)W∼N(0,nin​+nout​2​)$$

- ​ 是输入节点数
- noutn_{out}nout​ 是输出节点数
**适应场景:**
- `sigmoid`, `tanh`
- 不适用`ReLU`
