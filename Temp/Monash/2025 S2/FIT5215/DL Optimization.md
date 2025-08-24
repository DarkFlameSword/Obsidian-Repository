---
date: 2025-08-24
author:
  - Siyuan Liu
tags:
  - FIT5215
aliases:
  - summary
---
![[Pasted image 20250824192446.png]]
# Gradient vanishing
![[Pasted image 20250807174530.png]]
**定义:**
Gradients get smaller and smaller as the algorithm progresses down to the lower layers

**表现:**
- **Loss 曲线:**
    - 训练集 Loss 下降缓慢
    - 训练集 Loss 在后期几乎不再下降，趋于平缓
    - 验证集 Loss 也呈现相似的趋势
- **Accuracy 曲线:**
    - 训练集 Accuracy 提升缓慢
    - 训练集 Accuracy 在后期几乎不再提升，趋于平缓
    - 验证集 Accuracy 也呈现相似的趋势

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
![[Pasted image 20250807173729.png]]
**定义:**
The gradients can grow bigger and bigger, so many layers get insanely large weight updates, and the training diverges

**表现:**
- **Loss 曲线:**
    - 训练集 Loss 在初期迅速增加
    - 训练集 Loss 出现明显的震荡，可能包含 NaN 值
    - 验证集 Loss 也可能受到影响，出现震荡
- **Accuracy 曲线:**
    - 训练集 Accuracy 在初期迅速下降
    - 训练集 Accuracy 出现明显的震荡
    - 验证集 Accuracy 也可能受到影响，出现下降

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
## Xavier Weight Initialization
**作用:**
Try to ensure the variance of the outputs of each layer equal to the
variance of its input. This way, signals and gradients don't shrink or amplify layer by layer in the network

**计算步骤:**
假设某一层有：
- 输入单元数：n_{in}
- 输出单元数：n_{out}
权重矩阵 W 的元素希望满足：
$$Var(W x) \approx Var(x)$$

Xavier 初始化给出了一个简单公式：
- 对**均匀分布**：
$$W \sim U\left[
-\sqrt{\frac{6}{n_{in} + n_{out}}},\sqrt{\frac{6}{n_{in} + n_{out}}}
\right]$$

- 对**正态分布**：
$$W \sim N\left(0, \frac{2}{n_{in} + n_{out}}\right)$$

- ​ $n_{in}$是输入节点数
- $n_{out}$是输出节点数
**适应场景:**
- `sigmoid`, `tanh`
    - 因为这两个函数在输入较大时会饱和，容易导致梯度消失
- 不适用`ReLU`

![[Pasted image 20250824184323.png]]

## He Weight Initialization
**作用:**
Ensure the variance of the outputs of each layer equal to the variance of its inputs, but `He` specially optimized `ReLU`

**Why?:**
Xavier 初始化假设激活函数近似**线性**，但 ReLU 并非对称线性函数，特别是它会把负数全部置零，这会改变输出的方差。因此，需要针对 ReLU 设计新的初始化方式

**计算步骤:**
假设某一层有：
- 输入单元数：n_{in}
- 输出单元数：n_{out}
`ReLU` 的特点是：
$$\text{ReLU}(x) = \max(0,x)$$

大约 **一半的输入会被置为 0**，因此输出的方差会减半。为了保证输出的方差和输入相同，我们需要在初始化时把方差放大一点：
- 对**均匀分布**：
$$W \sim U\left[
-\sqrt{\frac{6}{n_{in}}},\sqrt{\frac{6}{n_{in}}}
\right]$$

- 对**正态分布**：
$$W \sim N\left(0, \frac{2}{n_{in}}\right)$$

- ​ $n_{in}$是输入节点数
- $n_{out}$是输出节点数
==分母是 2 倍，因为 `ReLU` 会丢掉一半的信号==

**适应场景:**
- `ReLU`, `ReLU的所有变种`
- 深层卷积神经网络 / 前馈网络 都可以用 He 初始化

![[Pasted image 20250824185407.png]]


# Overfitting
![[Pasted image 20250824192640.png]]
![[Pasted image 20250824192802.png]]


# Underfitting

# Regularization




# Ill-conditioning problem

# Long-term dependencies

# Poor correspondence between local and global structures

# Theoretical limits of optimization (but they usually have little use in practice of deep learning)


