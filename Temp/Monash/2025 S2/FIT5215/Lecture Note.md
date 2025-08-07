---
date: 2025-07-27
tags:
  - FIT5215
author:
  - Siyuan Liu
aliases:
  - note
---
# Vector

![[Pasted image 20250727164025.png]]
==Attention:==
1. 一般用变量表示的向量默认是列向量, 横向量需要使用转置符号`T`标明
## Multiplication

![[Pasted image 20250727164902.png]]
## Transpose
![[Pasted image 20250727165020.png]]
## p-norm/范数
![[Pasted image 20250727165647.png]]
### The Length of Vector
当p=2的时候也叫Frobenius范数, 一般我们求矩阵长度使用的就是该范数

### Distance between Two Vectors
![[Pasted image 20250727170036.png]]
### The Angel between Two Vectors
![[Pasted image 20250727170151.png]]
# Matrix 2D
![[Pasted image 20250727171416.png]]
==Attention==
1. AB矩阵相乘, 最后的结果矩阵的shape会取A的行数B的列数
2. 第一个矩阵 (A) 的列数必须等于第二个矩阵 (B) 的行数, 否则不能相乘
# AI Model Types

## Supervised Learning
**核心思想：** 通过**有答案的“练习题”**来学习。

把它想象成一个学生（机器学习模型）在学习。我们为他提供大量的练习题（输入数据 `X`），并且**每一道题都附带了标准答案**（输出标签 `y`）。学生的目标就是学习从题目到答案之间的规律和映射关系。

**关键特征：**

- **使用“已标记”的数据 (Labeled Data)**：数据集中的每个样本都包含两部分：**特征 (Features)** 和与之对应的 **标签 (Label)** 或 **目标 (Target)**。
- **目标**：学习一个函数 `f`，使得对于新的、从未见过的数据 `X_new`，模型能够预测出正确的标签 `y_new`，即 `y_new ≈ f(X_new)`。
### Classification
**核心思想：** 预测一个**离散的类别标签**。简单说，就是做“选择题”。

模型的任务是将输入数据分到一个预先定义好的类别中。输出的结果是有限的、不连续的类别。

**目标：** 输出是一个**类别 (Class)**。
#### 计算Loss的公式
1. **Mean Squared Error**
```
MSE = (1/n) * Σ(yᵢ - ŷᵢ)²
```
- `n`: 样本总数
- `yᵢ`: 第 `i` 个样本的真实值
- `ŷᵢ`: 第 `i` 个样本的预测值

==Attention:==
- 将这个误差**求平方**。这使得所有误差都为正数，并且会**极大地惩罚那些误差很大的预测**（比如真实值是10，你预测成100，平方后差距会变得非常大）

2. **Mean Absolute Error**
```
MAE = (1/n) * Σ|yᵢ - ŷᵢ|
```
==Attention:==
-  MAE 对所有大小的误差都给予相同的线性惩罚。它对**异常值 (Outliers)** 不像 MSE 那么敏感。如果你的数据中有很多离谱的异常点，使用 MAE 可能会让模型更稳定
### Regression
**核心思想：** 预测一个**连续的数值**。简单说，就是做“填空题”，填一个具体的数字。

模型的任务是基于输入数据，预测一个精确的、连续的输出值。

**目标：** 输出是一个**数值 (Value)**

#### 计算Loss的公式
1. **Cross-Entropy Loss**
```
BCE = - (1/n) * Σ [ yᵢ * log(ŷᵢ) + (1 - yᵢ) * log(1 - ŷᵢ) ]
```
- `yᵢ`: 真实标签，只能是 **0 或 1**。
- `ŷᵢ`: 模型预测的概率，通常是经过 `Sigmoid` 函数输出的，值在 (0, 1) 之间，表示样本为类别 1 的概率。

==Attention:==
- 如果真实标签 `y=1`: 公式简化为 `-log(ŷ)`。为了让 Loss 变小，`log(ŷ)` 就要变大，这意味着 `ŷ` 必须**趋近于 1
- 如果真实标签 `y=0`: 公式简化为 `-log(1 - ŷ)`。为了让 Loss 变小，`log(1 - ŷ)` 就要变大，这意味着 `(1 - ŷ)` 必须趋近于 1，也就是 `ŷ` 必须**趋近于 0**

1. **Categorical Cross-Entropy**
```
CCE = - (1/n) * Σ Σ [ yᵢ,c * log(ŷᵢ,c) ]
```
- `yᵢ,c`: 是一个 **One-Hot 编码**的向量。如果第 `i` 个样本的真实类别是 `c`，则 `yᵢ,c=1`，否则为 0。
- `ŷᵢ,c`: 模型预测的概率分布，通常是经过 `Softmax` 函数输出的，表示第 `i` 个样本属于类别 `c` 的概率。

==Attention:==
- 由于 One-Hot 编码的存在，对于每个样本 `i`，只有一个 `yᵢ,c` 是 1，其他都是 0。所以内层求和 `Σ` 会被简化
## Reinforce Learning

# Performance Metrics

### Accuracy
### Recall
### Precision
### F-score

# Feed-forward Neural Networks
```
import torch
import torch.nn as nn

# 定义一个前馈神经网络
class SimpleFFNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 4)   # 输入3个特征，隐藏层4个神经元
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)   # 输出1个结果

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 输入示例
input = torch.tensor([[1.0, 2.0, 3.0]])
model = SimpleFFNN()
output = model(input)
print(output)
```