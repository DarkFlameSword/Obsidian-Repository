---
date: 2026-03-11
author:
  - Siyuan Liu
tags:
  - FIT5221
---
# Edge Detection
For 2D function, $f(x,y)$, the partial derivative is:
$$\frac{\partial f(x,y)}{\partial x}= \lim_{\epsilon \rightarrow0}\frac{f(x+\epsilon,y)-f(x,y)}{\epsilon}$$
For discrete data, we can approximate using finite differences:
$$\frac{\partial f(x,y)}{\partial x}= \lim_{\epsilon \rightarrow0}\frac{f(x+\epsilon,y)-f(x,y)}{1}$$
![[Pasted image 20260311231829.png]]
![[Pasted image 20260325151059.png]]

The gradient of an image:
$$\nabla f=[\frac{\partial f}{\partial x},\frac{\partial f}{\partial y}]$$
The gradient direction of an image:
$$\theta =tan^{-1}\left[\frac{\frac{\partial f}{\partial x}}{\frac{\partial f}{\partial y}} \right]$$
The edge strength is given by the gradient magnitude:
$$||\nabla f||=\sqrt{(\frac{\partial f}{\partial x})^2+(\frac{\partial f}{\partial y})^2}$$

### Canny edge detector
**工作原理：**
1. 用高斯导数对图像进行滤波
先用Gaussian filter对图像进行平滑去噪，然后再对x与y轴分别求导

2. Find magnitude and orientation of gradient
得到 $G_x$ 和 $G_y$。
- **幅值**：代表边缘的强度，计算公式为 $\sqrt{G_x^2 + G_y^2}$。值越大，说明这里是边缘的可能性越高。
- **方向**：代表边缘的走向（垂直于梯度方向），计算公式为 $\theta = \arctan(G_y / G_x)$。

3. 非极大值抑制：将宽的“山脊”细化为单像素宽度

在计算完梯度幅值后，真实的边缘周围通常会形成一条比较宽的“高值带”（就像宽阔的山脊），而不是锐利的一条线。为了得到精确的边缘，我们需要进行“瘦身”

**操作方法**：算法会遍历图像中的每一个像素。对于当前像素，它会查看其**梯度方向**上的相邻像素。如果当前像素的梯度幅值**不是**这个方向上的局部最大值（即它比旁边的像素小/黑），那么就将它的值抑制为 0 黑色（当作非边缘处理）
![[Pasted image 20260312161343.png]]

**目的**：确保提取出的边缘是细致的、单像素宽的线条，精确定位物理边缘的中心

4. Linking and thresholding （hysteresis滞后）
经过上述处理后，我们得到了单像素宽的边缘候选点。传统方法是用一个单一阈值来判断：高于阈值就是边缘，低于就不是。但这容易导致真实的边缘断裂（因为某些边缘片段由于光照等原因幅值稍微变弱了一点）。Canny 算法引入了 **双阈值（滞后阈值）** 机制

Define two thresholds: low and high
- 高于**High threshold**的像素，被标记为“强边缘”，我们非常确定它们是真实的边缘
- 低于**Low threshold**的像素，直接被抛弃，视为噪声或非边缘
- 介于两者之间的像素，被称为**弱边缘**，它们处于待定状态

Linking：用高阈值开始边缘曲线，用低阈值延续它们

算法从“强边缘”像素开始追踪。如果一个“弱边缘”像素在空间上与“强边缘”相连通，那么它就被“拯救”了，正式成为真实边缘的一部分。如果不与任何强边缘相连，则被当作噪声抛弃

**目的**：这种机制既能通过高阈值保证边缘的可靠性，又能通过低阈值和连通性分析保证边缘的连续性，有效解决了边缘断裂的问题

---
# How to find the edge?
### For correlation
![[Pasted image 20260312140500.png]]关于 $x$ 的偏导（或在 1D 切片上的导数 $\frac{d}{dx}f(x)$）在实际物理和图像意义上，代表着**像素强度（灰度值）在水平方向上的变化率**

---
### For Convolution
![[Pasted image 20260312143642.png]]

---
# Harris Corner Detection
![[Pasted image 20260312162223.png]]
- 基本原理：使用一个很小的滑动窗口，把这个小框向各个方向（上下左右、斜向）稍微移动一点点，如果在任何方向上移动，框内的图像内容都发生了剧烈变化，那么这个框所在的位置就是一个“角点”
$$E(u,v) = \sum_{x,y}w(x,y)[I(x+u,y+v)-I(x,y)]^2$$
- **$E(u,v)$**：代表Error变化量。它衡量的是窗口移动了 $(u,v)$ 距离后，窗口内图像整体的变化程度。我们希望找到让 $E(u,v)$ 很大的位置
- **$(u,v)$**：代表窗口的**Shift**。$u$ 是水平移动距离，$v$ 是垂直移动距离
- **$(x,y)$**：代表窗口内的**每一个像素点的坐标**
- **$\sum_{x,y}$**：代表将窗口内所有像素点的变化情况累加起来
- **$w(x,y)$**：叫做窗口函数或权重函数。它决定了窗口的大小和形状。最简单的是矩形窗口（框内权重为 1，框外为 0），更常用的是高斯窗口（越靠近中心权重越大，越平滑）
- **$I(x,y)$**：代表移动前原位置像素的**灰度值（Intensity）**
- **$I(x+u,y+v)$**：代表窗口移动 $(u,v)$ 之后，新位置像素的**灰度值**

Harris 角点检测公式 $E(u,v) = \sum [I(x+u,y+v)-I(x,y)]^2$ 中，当移动的步长 $(u,v)$ 非常小时，我们可以用**二维泰勒展开式**（只取一阶导数，忽略更高阶的极小项）来近似平移后的图像：

$$
\begin{align}
& I(x+u, y+v) \approx I(x,y) + u\cdot \frac{\partial I}{\partial x}(x,y) + v\cdot \frac{\partial I}{\partial y}(x,y)
\end{align}
$$
这里的 $I_x$ 和 $I_y$ 就是图像在水平和垂直方向的偏导数（梯度）。把 $I(x+u, y+v)$ 近似结果代回原来的误差公式
$$
\begin{align}
E(u,v) & = \sum_{x,y}w(x,y)[ I(x,y) + u\cdot \frac{\partial I}{\partial x}(x,y) + v\cdot \frac{\partial I}{\partial y}(x,y)  -I(x,y)]^2\\
& = \sum_{x,y}w(x,y)\left[ u^2\cdot (\frac{\partial I}{\partial x})^2 + 2u\cdot v\cdot \frac{\partial I}{\partial x}\cdot \frac{\partial I}{\partial y}+ v^2\cdot (\frac{\partial I}{\partial y})^2 \right]\\
& = \sum_{x,y}w(x,y) \cdot [u\;v]
\cdot
\begin{bmatrix} 
(\frac{\partial I}{\partial x})^2 
& \frac{\partial I}{\partial x}\cdot \frac{\partial I}{\partial y} \\
\frac{\partial I}{\partial x}\cdot \frac{\partial I}{\partial y}
&(\frac{\partial I}{\partial y})^2 
\end{bmatrix}
\cdot
\begin{bmatrix} 
u \\ v
\end{bmatrix}
\end{align}
$$
原本复杂的平方差就会巧妙地化简成一个由梯度组成的 $2 \times 2$ 矩阵（结构张量M）

$$
\begin{align}
M = & \sum_{x,y}w(x,y) \cdot 
\begin{bmatrix} 
(\frac{\partial I}{\partial x})^2 
& \frac{\partial I}{\partial x}\cdot \frac{\partial I}{\partial y} \\
\frac{\partial I}{\partial x}\cdot \frac{\partial I}{\partial y}
&(\frac{\partial I}{\partial y})^2 \\
\end{bmatrix}\\
& \sum_{x,y}w(x,y) 
\begin{bmatrix}
\frac{\partial I}{\partial x} \\ \frac{\partial I}{\partial y}
\end{bmatrix}
\begin{bmatrix}
\frac{\partial I}{\partial x} & \frac{\partial I}{\partial y}
\end{bmatrix}
\end{align}
$$
![[Pasted image 20260312170045.png]]
### 二次矩阵的含义
![[Pasted image 20260312170904.png]]

![[Pasted image 20260312170926.png]]
### 判断当前窗口里装的是什么地形（平坦区域、边缘、还是角点）

##### Second Moment Matrix（结构张量）
![[Pasted image 20260312182958.png]]

$\begin{bmatrix} u & v \end{bmatrix} M \begin{bmatrix} u \\ v \end{bmatrix} = \text{const}$ 表示 **在各个方向上，我们需要走多远，才能让图像的变化量达到一个固定的常数？”** 这个走出来的轨迹，在数学上就是一个椭圆

**矩阵对角化与特征值（Eigenvalues）：** 现实中的边缘可能是倾斜的，为了方便研究，我们用线性代数里的“对角化（Diagonalization）”把坐标系旋转到刚好对齐这个椭圆的长短轴（旋转矩阵 $R$）。 提取出来的 $\lambda_1$ 和 $\lambda_2$ 就是矩阵 $M$ 的**特征值**。它们代表了在这个窗口里，**变化最快**和**变化最慢**的两个正交方向的“梯度能量“

##### 特征值给图像像素可视化分类
![[Pasted image 20260312215924.png]]
- **“Flat” region（平坦区域 - 灰色左下角）：**
    如果 $\lambda_1$ 和 $\lambda_2$ **都很小**
    - **物理意义：** 你往任何方向滑动窗口，图像都没什么变化
    - **场景：** 比如一张均匀的白纸，或者 CT 扫描中一块密度完全均匀的组织内部

- **“Edge”（边缘 - 橙色区域）：**
    如果其中一个特征值很大，另一个很小（$\lambda_1 \gg \lambda_2$ 或 $\lambda_2 \gg \lambda_1$）
    - **物理意义：** 图像在一个方向上变化极其剧烈（跨越了边界），但在垂直的另一个方向上毫无变化（沿着边界滑动）
    - **场景：** 比如桌子的边缘，或者眼底图像中一根笔直的微小血管的血管壁。你垂直于血管走，颜色突变；顺着血管走，颜色不变

- **“Corner”（角点 - 蓝色右上角）：**
    如果 $\lambda_1$ 和 $\lambda_2$ **都非常大**，且数值相近（$\lambda_1 \sim \lambda_2$）
    - **物理意义：** 无论你往哪个方向移动哪怕一点点，图像内容都会发生巨大的改变
    - **场景：** 比如棋盘格的十字交叉点，或者微血管网络中的**血管分叉点（Bifurcations）**。这种点在图像中具有极高的独特性，是计算机视觉做图像匹配、追踪和 3D 重建时的“圣杯”


---

# Feature Points
## Panorama Stitching全景拼接
![[Pasted image 20260312161849.png]]
1.  Detect keypoints and compute features for keypoints in both images
![[Pasted image 20260312161900.png]]
2. Find corresponding pairs (by matching keypoint features between two images)

![[Pasted image 20260312161928.png]]
3. Use these pairs to align the images

---
# 2D高斯filter的二阶偏导
![[Pasted image 20260312144408.png]]

---

# Derivative Filters

### Prewitt
### Sobel
### Roberts

---



