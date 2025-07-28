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
# Kullback-Leibler (KL) divergence
个人理解: KL散度用来计算, 当前模型距离真实案例的偏差值

**D_KL(P || Q) = Σ P(x) * log₂( P(x) / Q(x) )**
==![[Pasted image 20250728164017.png]]==

![[Pasted image 20250727172836.png]]
# Entropy of the distribution
个人理解: 衡量的是一个随机事件或一个概率分布的**不确定性**或**混乱程度**, 也就是要搞清楚一个随机事件的最终结果，平均需要多少信息量（通常用比特`bit`来衡量）
- **熵越高**，代表系统越混乱，结果越不可预测
- **熵越低**，代表系统越有序，结果越容易预测

**H(X) = - Σ p(x) * log₂(p(x))**
==![[Pasted image 20250728164017.png]]

![[Pasted image 20250727173520.png]]
# Cross-entropy (CE)
个人理解: 计算预测结果与真实结果之间的差距, CE越大预测结果越离谱

**H(p, q) = - Σ p(x) * log₂(q(x))**
==![[Pasted image 20250728164017.png]]

![[Pasted image 20250727174454.png]]
==Attention:==
1. CrossEntropy(p, q) = Entropy(p) + KL_Divergence(p || q) **在机器学习中, 真实数据分布 `p` 是固定的, 所以 `Entropy(p)` 是一个常数, 所以最小化交叉熵就等价于最小化KL散度**

# `Softmax`


# AI Model Types
### Classification
### Regression
### Supervised Learning

### Reinforce Learning

