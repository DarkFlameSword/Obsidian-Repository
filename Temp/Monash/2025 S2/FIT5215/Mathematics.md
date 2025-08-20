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
==当一个函数的输入和输出都是多维向量时，它的“导数”不再是一个简单的数字（斜率）或向量（梯度），而是一个**矩阵**，这个矩阵被称为**雅可比矩阵 (Jacobian Matrix)**==

$$𝑓: ℝ^𝑚 → ℝ^n$$
- **输入**：是一个 $m$ 维的向量 $𝑥 = (𝑥₁, … , 𝑥ₘ)$, 它属于 $m$ 维实数空间 $ℝ^m$
- **输出**：是一个 $n$ 维的向量 $𝑦 = (𝑦₁, … , 𝑦ₙ)$, 它属于 $n$维实数空间 $ℝ^n$

这个函数$f$实际上是$n$个独立函数的集合，每个函数$f_i$都接收 $m$维的输入$x$，并各自产生一个实数输出$y_i: y₁ = f_1(x_1,\; \dots ,\; x_m) \quad y₂ = f_2(x_1,\; \dots ,\; x_m) \quad... \quad y_n = f_n(x_1,\; \dots ,\; x_m)$

## The Jacobian Matrix 雅可比矩阵

在某个点 `a` 的导数$∇f(a)$ / $\frac{∂x}{∂y}f(a)$是一个 $n \times m$ 的矩阵。它的结构如下：

- **矩阵的每一行** 对应一个输出函数 $f_i$
- **矩阵的每一列** 对应一个输入变量 $x_j$
- 矩阵中第 $i$ 行、第 $j$ 列的元素是 $\frac{∂f_i} {∂x_j}$，即**第 $i$ 个输出函数**相对于**第 $j$ 个输入变量**的偏导数。
- 所有这些偏导数都在点 $a$ 进行求值

$$ \frac{\partial y}{\partial x}(a) = \nabla f(a) = 
\overset{\color{red}m}
{\begin{bmatrix}
\frac{\partial f_1}{\partial x_1}(a) & \cdots & \frac{\partial f_1}{\partial x_j}(a) & \cdots & \frac{\partial f_1}{\partial x_m}(a) \\
\vdots & \ddots & \vdots & \ddots & \vdots \\
\frac{\partial f_i}{\partial x_1}(a) & \cdots & \frac{\partial f_i}{\partial x_j}(a) & \cdots & \frac{\partial f_i}{\partial x_m}(a) \\
\vdots & \ddots & \vdots & \ddots & \vdots \\
\frac{\partial f_n}{\partial x_1}(a) & \cdots & \frac{\partial f_n}{\partial x_j}(a) & \cdots & \frac{\partial f_n}{\partial x_m}(a)
\end{bmatrix}
}\rlap{\quad \color{red}n}$$


### 例子:
$$
y = f(x) = f(x_1, x_2, x_3) = (x_1^2 + x_2^2, x_2^2 + x_3^2x_2)
$$
- $f: \mathbb{R}^3 \to \mathbb{R}^2$
- $f_1(x) = f_1(x_1, x_2, x_3) = x_1^2 + x_2^2$
- $f_2(x) = f_2(x_1, x_2, x_3) = x_2^2 + x_3^2x_2$
- $\frac{\partial y}{\partial x} = \nabla f \in \mathbb{R}^{2 \times 3}$
$${\color{red}\frac{\partial y}{\partial x}} = {\color{red}\nabla_x f} = 
{\color{green}
\begin{bmatrix} 
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \frac{\partial f_1}{\partial x_3} \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \frac{\partial f_2}{\partial x_3} 
\end{bmatrix}}
=
\begin{bmatrix} 
2x_1 & 2x_2 & 0 \\
0 & 2x_2 + x_3^2 & 2x_2x_3
\end{bmatrix}$$
# Probabilistic
## **1. PDF：Probability Density Function（概率密度函数）**

### 定义

- 对连续随机变量 XXX，PDF 是一个函数 fX(x)f_X(x)fX​(x)，满足：$fX(x)≥0∀xf_X(x) \ge 0 \quad \forall xfX​(x)≥0∀x ∫−∞∞fX(x) dx=1\int_{-\infty}^{\infty} f_X(x) \, dx = 1∫−∞∞​fX​(x)dx=1$

### 含义

- fX(x)f_X(x)fX​(x) 本身 **不是概率**，而是概率密度。
    
- 在一个小区间 [a,b][a, b][a,b] 上的概率可以通过积分得到：
    

P(a≤X≤b)=∫abfX(x) dxP(a \le X \le b) = \int_a^b f_X(x) \, dxP(a≤X≤b)=∫ab​fX​(x)dx

### 举例

- 标准正态分布：
    

fX(x)=12πe−x2/2f_X(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}fX​(x)=2π​1​e−x2/2

---

## **2. CDF：Cumulative Distribution Function（累积分布函数）**

### 定义

- 对连续随机变量 XXX，CDF 是：
    

FX(x)=P(X≤x)=∫−∞xfX(t) dtF_X(x) = P(X \le x) = \int_{-\infty}^{x} f_X(t) \, dtFX​(x)=P(X≤x)=∫−∞x​fX​(t)dt

### 含义

- 给定一个 xxx，CDF 告诉你 **随机变量小于等于 x 的概率**。
    
- 单调递增，范围从 0 到 1：
    

FX(−∞)=0,FX(∞)=1F_X(-\infty) = 0, \quad F_X(\infty) = 1FX​(−∞)=0,FX​(∞)=1

### 举例

- 标准正态分布的 CDF：
    

FX(x)=∫−∞x12πe−t2/2dtF_X(x) = \int_{-\infty}^{x} \frac{1}{\sqrt{2\pi}} e^{-t^2/2} dtFX​(x)=∫−∞x​2π​1​e−t2/2dt

---

## **3. PDF 与 CDF 的关系**

FX(x)=∫−∞xfX(t) dtF_X(x) = \int_{-\infty}^{x} f_X(t) \, dtFX​(x)=∫−∞x​fX​(t)dt fX(x)=ddxFX(x)f_X(x) = \frac{d}{dx} F_X(x)fX​(x)=dxd​FX​(x)

- **CDF 是 PDF 的积分**
    
- **PDF 是 CDF 的导数**
    

---

## **4. 直观理解**

- **PDF**：描述“每个点附近概率的密集程度”，类似曲线高度
    
- **CDF**：描述“累积概率”，类似曲线从 0 到 1 逐渐爬升
    

---

## **5. 图像示意**

- PDF 曲线：钟形（标准正态）
    
- CDF 曲线：S 形，从 0 慢慢上升到 1