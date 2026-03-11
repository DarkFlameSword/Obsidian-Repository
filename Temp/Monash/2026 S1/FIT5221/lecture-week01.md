---
date: 2026-03-05
author:
  - Siyuan Liu
tags:
  - FIT5221
---
# Task
## Classification
- Input: Image
- Output: class label
![[Pasted image 20260311182817.png]]
---
## Object Detection
- Input: image
- Output: object bounding boxes and class labels
![[Pasted image 20260305191343.png]]
---
## Semantic Segmentation
- Input: image
- Output: class label for each pixel
![[Pasted image 20260305191513.png]]
---
## Instance Segmentation
- Input: image
- Output: a segmentation mask with class label for each object instance
![[Pasted image 20260305191539.png]]

---
# Two dominated styles used in CV
# Grayscale images
---
# RGB color images
- Each RGB color image has 3 numbers (R,G,B) per pixel
- Usually, R, G & B values are in [0, 255]
---
# Point operations
The Definition: Applying operations on pixels individually

$$g(x,y) = g \cdot f(x,y)+b$$
- $g:$ gain(control contrast)
- $b:$ bias(control brightness)
![[Pasted image 20260311215026.png]]
---

# Image histogram

![[Pasted image 20260311215105.png]]- Histogram captures the distribution of gray levels in the image.
- How frequently each gray level occurs in the image
---
# Image Filtering
在图像处理中，卷积核就像是一个**刻着特定图案的模具**或**印章**。

卷积的过程，就是拿着这个模具，在原始图像上从左到右、从上到下一点点滑动。每滑动到一个位置，就对比一下：**“底下的图像长得像不像我手里的模具？”**

- 如果长得像，计算出来的结果数值就**很大**（特征被激活）
- 如果长得不像或者毫无关系，结果数值就**接近 0**

用途：
- Enhance an image (denoise, resize, etc)
- Extract information (texture, edges, etc)
- Detect patterns (template matching)
### Correlation
$$
\begin{align}
&G[i,j] = \sum_{u=-k}^{k}\sum_{v=-k}^{k} \; H[u,v] \; F[i+u,j+v] \\
&\text{equals to} \to G = H \otimes F
\end{align}
$$
- $H[u,v]:$ kernel
##### $\otimes$ Kronecker product

$$
\begin{bmatrix} 1 & 2 \\ 3 & 1 \end{bmatrix} \otimes \begin{bmatrix} 0 & 3 \\ 2 & 1 \end{bmatrix} = \begin{bmatrix} 1 \cdot 0 & 1 \cdot 3 & 2 \cdot 0 & 2 \cdot 3 \\ 1 \cdot 2 & 1 \cdot 1 & 2 \cdot 2 & 2 \cdot 1 \\ 3 \cdot 0 & 3 \cdot 3 & 1 \cdot 0 & 1 \cdot 3 \\ 3 \cdot 2 & 3 \cdot 1 & 1 \cdot 2 & 1 \cdot 1 \end{bmatrix} = \begin{bmatrix} 0 & 3 & 0 & 6 \\ 2 & 1 & 4 & 2 \\ 0 & 9 & 0 & 3 \\ 6 & 3 & 2 & 1 \end{bmatrix}
$$---
### Convolution
##### 为什么要反转filter？
这源于传统的信号与系统理论。翻转是为了保证卷积运算满足**结合律**：

$$(Image * Filter_1) * Filter_2 = Image * (Filter_1 * Filter_2)$$

这意味着如果我们要对图像连续做两次复杂的滤波，我们可以**先在纸上把两个滤波器通过卷积合并成一个超级滤波器**，然后再去和图像运算。这能省下海量的计算时间！如果不翻转（也就是做 Correlation），结合律是不成立的

$$
\begin{align}
&G[i,j] = \sum_{u=-k}^{k}\sum_{v=-k}^{k} \; H[u,v] \; F[i-u,j-v] \\
&\text{equals to} \to G = H \star F
\end{align}
$$
- 相当于把filter上下左右全部颠倒之后做correlation
![[Pasted image 20260311223034.png]]
- convolution满足交换律，分配律，结合律

---

## Separable Kernel
- 如果一个 2D 卷积核允许我们这样做：即拆分后两次一维卷积的结果，与直接进行一次二维卷积的结果在数学上绝对相等
- 一个二维矩阵如果能被称为“可分离”的，那么它必定可以由一个列向量乘以一个行向量来精确生成
$$
\begin{bmatrix}
1&\\
1&\\
1&\\
\end{bmatrix}
\times
\begin{bmatrix}
1& 1& 1\\
\end{bmatrix}
=
\begin{bmatrix}
1& 1& 1\\
1& 1& 1\\
1& 1& 1\\
\end{bmatrix}
$$

### Box Filter（Mean Filter）
![[Pasted image 20260311221626.png]]
- 通常为奇数长度的卷积核
- 将卷积核中的每一个值与对应图像的灰度值相乘，然后卷积核范围内的所有结果相加，和的结果赋值给中心位置
- **优点（去噪与平滑）：** 作为一个快速的预处理步骤，Box Filter 可以迅速压制那些突兀的孤立噪点，让整体图像的灰度过渡更加平滑
- **致命缺点（边缘模糊）：** 因为它对窗口内的所有像素“一视同仁”，所以它无法区分哪里是背景噪点，哪里是重要的结构边缘。在求平均值的过程中，原本锐利的边缘（比如组织的边界）会被无情地模糊掉
### Gaussian Filter
![[Pasted image 20260311223208.png]]
$$G_{\sigma} = \frac{1}{2\pi\sigma^{2}}\;e^{-\frac{x^2+y^2}{2\sigma^2}}$$
- $x$ 和 $y$ 是卷积核中某个点到中心点的距离
- $\sigma$是标准差，它决定了这口“钟”是又高又瘦，还是又矮又胖。$\sigma$ 越大，平滑（模糊）的效果就越强烈

**应用注意点：**
- 注意filter的size选取
- filter的边缘应该接近0
- filter的边长的一半接近$3\sigma$效果最好

**特性：**
- Gaussian ffilter是低通滤波器
- 两个正态分布的卷积，其结果必定是一个更宽的正态分布。无论你怎么卷积，它的形状永远保持那条完美的“钟形曲线”，绝对不会变形或者产生奇怪的涟漪（如果你想要一个非常强烈、范围很大的模糊效果
	即需要一个很大的 $\sigma$ 值和巨大的卷积核，比如 31x31，你不需要真的去创建一个那么庞大且计算缓慢的卷积核。你完全可以拿一个很小的卷积核（比如 3x3，较小的 $\sigma$），在图像上连续过滤多次。最终得到的平滑效果，和直接用那个巨大的卷积核是一模一样的
- 两个高斯函数发生卷积时，它们合并后的新高斯函数，其Variance，即 $\sigma^2$是相加的
	假设我们用同一个标准差为 $\sigma$ 的高斯核对图像连续卷积两次：
	1. 第一次卷积后的方差：$\sigma^2$
	2. 第二次卷积后的总方差：$\sigma_{total}^2 = \sigma^2 + \sigma^2 = 2\sigma^2$
	因为标准差是方差的平方根，所以合并后的总标准差 $\sigma_{total}$ 就是：
$$\sigma_{total} = \sqrt{2\sigma^2} = \sigma\sqrt{2}$$
- Gaussian kernel is separable: it factors into product of two 1D Gaussians
	从二维高斯公式就能看出端倪（指数相加等于底数相乘）：
$$e^{-\frac{x^2+y^2}{2\sigma^2}} = e^{-\frac{x^2}{2\sigma^2}} \cdot e^{-\frac{y^2}{2\sigma^2}}$$
**根据Gaussian filter的可分解性（separable）：**
当我们使用一个标准的二维卷积核（假设它的宽和高都是 $K$，也就是一个 $K \times K$ 的正方形矩阵）去处理图像时，为了计算出输出图像中**仅仅一个像素**的值，我们需要把这个卷积核完全覆盖在输入图像上

我们要把这 $K \times K$ 个格子里的每一个数字，都与它底下对应的图像像素进行“乘法”操作，然后再把所有的结果“加”起来。这个“multiply-add”的操作一共要进行 $K \times K = K^2$ 次

假如拆分成两步连续的一维卷积：
1. 先用一个 $1 \times K$ 的水平卷积核（只包含一行），横向扫过整张图像。对于每个像素，这需要 $K$ 次运算
2. 拿着第一步生成的“半成品”图像，再用一个 $K \times 1$ 的垂直卷积核（只包含一列），纵向扫过。这又需要 $K$ 次运算
两次加起来，为了得到最终这一个像素的值，总共只需要 $K + K = 2K$ 次运算


