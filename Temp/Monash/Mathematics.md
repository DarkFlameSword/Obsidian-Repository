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

^42723b

![[Pasted image 20250727164025.png]]
==Attention:==
1. 一般用变量表示的向量默认是列向量, 横向量需要使用转置符号`T`标明
## Multiplication

![[Pasted image 20250727164902.png]]

## @ Scalar Product
符号 @ 表示矩阵乘法，不是逐元素相乘。它是 Python 3.5+ 中用于线性代数乘法的运算符（相当于 np.matmul / torch.matmul）

**举例：**
$$\begin{bmatrix}
1&0&2\\-1&3&1
\end{bmatrix}\times
\begin{bmatrix}
3&1\\2&1\\1&0
\end{bmatrix}=
\begin{bmatrix}
5&1\\4&2
\end{bmatrix}
$$

## $\odot$ Hadamard Product
符号 $\odot$ 表示逐元素相乘

![[Pasted image 20251019145012.png]]


## $\otimes$ Kronecker product

$$
\begin{bmatrix} 1 & 2 \\ 3 & 1 \end{bmatrix} \otimes \begin{bmatrix} 0 & 3 \\ 2 & 1 \end{bmatrix} = \begin{bmatrix} 1 \cdot 0 & 1 \cdot 3 & 2 \cdot 0 & 2 \cdot 3 \\ 1 \cdot 2 & 1 \cdot 1 & 2 \cdot 2 & 2 \cdot 1 \\ 3 \cdot 0 & 3 \cdot 3 & 1 \cdot 0 & 1 \cdot 3 \\ 3 \cdot 2 & 3 \cdot 1 & 1 \cdot 2 & 1 \cdot 1 \end{bmatrix} = \begin{bmatrix} 0 & 3 & 0 & 6 \\ 2 & 1 & 4 & 2 \\ 0 & 9 & 0 & 3 \\ 6 & 3 & 2 & 1 \end{bmatrix}
$$

---
## Transpose
![[Pasted image 20250727165020.png]]
## p-norm/范数

^839d47

![[Pasted image 20250727165647.png]]
### The Length of Vector

^710368

当p=2的时候也叫Frobenius范数, 一般我们求矩阵长度使用的就是该范数

### Distance between Two Vectors
![[Pasted image 20250727170036.png]]
### The Angel between Two Vectors
![[Pasted image 20250727170151.png]]
==Cosine distance can be computed via Euclidean distance if vectors are made unit vectors:==
$$d_{cosine}​(x,y)= 1−\cos\theta = \frac{1}{2}||\hat{x}−\hat{y}||^2$$
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
## Probability Density Function(PDF)

**定义:**
对连续随机变量 $X$，PDF 是一个函数 $f_X(x)$，满足：
$$ \forall x \quad f_X(x)≥0$$
$$\forall x \quad \int_{-\infty}^{\infty} f_X(x)dx=1$$
$$P(a\le X \le b) = \int_{a}^{b} f_X(x)dx$$
**理解:**
PDF: 给定一个 $x$，PDF 告诉你随机变量在$x$ 附近取值的“概率密度”，也就是概率在数轴上的分布浓度，只有把它在区间上积分，才得到真实的概率

---
## Cumulative Distribution Function(CDF)

**定义:**
对连续随机变量 X，CDF 是：
$$F_X(x)=P(X \le x)$$
CDF满足:
$$F(x_1)\le F(x_2),\quad x_1<x_2$$
$$\begin{align} \lim_{x \to - \infty} F(x) =0 \\ \lim_{x \to \infty} F(x) =1 \end{align}$$
$$$$
$$\begin{align} f(x) = \frac{d}{dx} F(x) \\ F(x) = \int_{-\infty}^{x} f_X(t) dt \end{align}$$

**理解:**
给定一个 $x$，CDF 告诉你随机变量小于等于 x 的概率

---
# Eigenvalue, Eigenvector, Eigenmatrix

# Diagonalizable Matrix

当输入是向量时，`diag()` 创建一个对角矩阵，向量元素作为对角线元素：
$$\begin{aligned}
&σ'(h¹) = [a, b, c]\\
&diag(σ'(h¹)) = 
\begin{bmatrix}
a& 0& 0\\
0& b& 0\\
0& 0& c\\
\end{bmatrix}
\end{aligned}$$

---

# Taylor series expansion
**核心思想**：用简单的多项式去无限逼近一个复杂的函数

$$f(x) = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \frac{f'''(a)}{3!}(x-a)^3 + \dots + \frac{f^{(n)}(a)}{n!}(x-a)^n + \dots$$
- $f^{(n)}(a)$ 代表函数在 $a$ 点的 $n$ 阶导数
- $n!$ 是阶乘（例如 $3! = 3 × 2 × 1$）
- 当我们在 $a=0$ 处展开时，它有一个特殊的名字叫**麦克劳林级数（Maclaurin series）**

### 例子 A：自然指数函数 $e^x$ （最简单的展开）

指数函数 $e^x$ 有个神奇的特性：无论求多少次导数，结果永远是 $e^x$。

如果在 $a=0$ 处展开，因为 $e^0 = 1$，所以它的所有阶导数在这个点的值都是 **1**。

套入公式，我们就得到了极其干净的展开式：

$$e^x = 1 + x + \frac{x^2}{2} + \frac{x^3}{6} + \frac{x^4}{24} + \dots$$

### 例子 B：正弦函数 $\sin(x)$

如果在 $a=0$ 处对 $\sin(x)$ 展开。它的导数会在 $\sin$ 和 $\cos$ 之间循环，代入 $a=0$ 后，偶数阶导数全为 0，奇数阶导数为正负交替的 1。展开式如下：

$$\sin(x) = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \frac{x^7}{7!} + \dots$$

如上面的图示所表达的那样：只取第一项 $x$ 就是一条直线；取到前两项就是一个向下的曲线；取的项数越多，这个多项式就越像波浪形的 $\sin(x)$ 曲线

