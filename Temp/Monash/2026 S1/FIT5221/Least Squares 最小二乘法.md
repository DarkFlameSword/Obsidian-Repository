---
date: 2026-03-25
author:
  - Siyuan Liu
tags:
  - summary
---
最小二乘法（Least Squares）是"当方程比未知数多，找一个让所有方程'尽量满足'的最优解"。

先从一个具体问题建立直觉，再推导公式，最后连接到图像拼接。---

## 从"拟合一条直线"建立直觉

最经典的例子：给你 $n$ 个带噪声的测量点 $(x_i, y_i)$，找一条直线 $y = ax + b$ 最好地穿过它们。红色竖线就是每个点的**残差** $r_i = y_i - \hat{y}_i$。最小二乘要最小化的是残差的**平方和** $\text{SSE} = \sum r_i^2$，而不是绝对值之和——因为平方可微，方便求解析解。

![[Pasted image 20260325181038.png]]

---

## 矩阵形式推导

$n$ 个点，每个满足 $y_i \approx ax_i + b$，写成矩阵：

$$\underbrace{\begin{bmatrix} x_1 & 1 \ x_2 & 1 \ \vdots & \vdots \ x_n & 1 \end{bmatrix}}_{A \ (n\times 2)} \underbrace{\begin{bmatrix} a \ b \end{bmatrix}}_{\mathbf{x}} = \underbrace{\begin{bmatrix} y_1 \ y_2 \ \vdots \ y_n \end{bmatrix}}_{\mathbf{b}}$$

目标：最小化 $|\mathbf{b} - A\mathbf{x}|^2$。对 $\mathbf{x}$ 求导并令其为零：

$$\frac{\partial}{\partial \mathbf{x}}|A\mathbf{x} - \mathbf{b}|^2 = 2A^\top(A\mathbf{x} - \mathbf{b}) = \mathbf{0}$$

得到**法方程（Normal Equations）**：

$$\boxed{A^\top A , \mathbf{x}^* = A^\top \mathbf{b}}$$

$$\mathbf{x}^* = (A^\top A)^{-1} A^\top \mathbf{b}$$

其中 $(A^\top A)^{-1} A^\top$ 叫做 $A$ 的**伪逆**（Moore-Penrose Pseudoinverse），记作 $A^+$。

---

## 几何解释：
![[Pasted image 20260325180958.png]]

---

## 加权最小二乘：不同数据点的可信度不同

有时并非所有点同等可信——测量精度不同，或某些点是近似值。这时用**加权最小二乘（WLS）**：

$$\min_{\mathbf{x}} \sum_i w_i r_i^2 = \min_{\mathbf{x}} |W^{1/2}(A\mathbf{x} - \mathbf{b})|^2$$

解为：$\mathbf{x}^* = (A^\top W A)^{-1} A^\top W \mathbf{b}$

其中 $W = \text{diag}(w_1, \cdots, w_n)$，权重 $w_i$ 越大表示该点越可信。点击图中的数据点可以切换它的权重。可以看到：普通最小二乘（蓝线）会被异常点强烈拉偏；加权最小二乘（橙线）把可疑点的权重降低后，拟合结果更接近真实趋势。

---

## 回到图像拼接：单应矩阵的精化

RANSAC 找到内点集合之后，需要用所有内点对 $H$ 做一次最小二乘精化。每一对匹配点 $(x_i, y_i) \leftrightarrow (x_i', y_i')$ 贡献两行方程（DLT 算法）：

$$\begin{bmatrix} -x_i & -y_i & -1 & 0 & 0 & 0 & x_i'x_i & x_i'y_i & x_i' \ 0 & 0 & 0 & -x_i & -y_i & -1 & y_i'x_i & y_i'y_i & y_i' \end{bmatrix} \mathbf{h} = \mathbf{0}$$

$n$ 对点拼成 $2n \times 9$ 的矩阵 $A$，解 $A\mathbf{h} = \mathbf{0}$ 等价于求 $A^\top A$ 的最小特征值对应的特征向量——这正是 SVD（奇异值分解）最后一行的结果。

$$H^* = \text{reshape}\left(\arg\min_{|\mathbf{h}|=1} |A\mathbf{h}|^2\right) = \text{reshape}(V_{:,8})$$

其中 $V$ 是 $A = U\Sigma V^\top$ 中的右奇异向量矩阵，取最后一列。内点越多，$H$ 的估计越稳定——这就是为什么 RANSAC 结束后要用全部内点重新做这一步精化，而不是直接用最初那 4 个点的解。