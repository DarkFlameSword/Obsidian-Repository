---
date: 2025-08-08
author:
  - Siyuan Liu
tags:
  - FIT5215
aliases:
  - base
---
# Formulation
1. 导数乘法法则
$$(uv)' = u'v + uv'$$
2. 导数除法法则
$$\left(\frac{u}{v}\right)' = \frac{u'v - uv'}{v^2}$$
3. 指数函数求导
$$(e^u)' = u'e^u$$
4. 对数函数求导
$$(\log u)' = \frac{u'}{u}$$
5. Chain Rule
$$\frac{\partial u}{\partial x} = \frac{\partial u}{\partial v} \times \frac{\partial v}{\partial x}$$
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

# Derivative for multi-variate functions
Given a function 𝑓: $ℝ^𝑚 → ℝ^n$
𝑓 𝑥 = (𝑓1 𝑥 , … , 𝑓𝑛(𝑥)) where 𝑓1, … , 𝑓𝑛: ℝ𝑚 →

ℝ and 𝑥 = (𝑥1, … , 𝑥𝑚). Let denote 𝑦 = 𝑓(𝑥).