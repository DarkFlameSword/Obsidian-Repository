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

---

# Early Stopping

# Regularization


# Ill-conditioning problem

# Long-term dependencies

# Poor correspondence between local and global structures

# Theoretical limits of optimization (but they usually have little use in practice of deep learning)


