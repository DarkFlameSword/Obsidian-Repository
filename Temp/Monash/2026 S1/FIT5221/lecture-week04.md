---
date: 2026-03-25
author:
  - Siyuan Liu
tags:
  - FIT5221
---
# Basic 2D Transformations

![[Pasted image 20260325182835.png]]

### 一、平移（Translation）

最简单的变换，只移动位置，不改变任何形状属性。

$$T = \begin{bmatrix} 1 & 0 & t_x \ 0 & 1 & t_y \ 0 & 0 & 1 \end{bmatrix}$$

$t_x$、$t_y$ 直接藏在第三列。这正是齐次坐标最原始的动机——普通 $2\times2$ 矩阵无法表示平移，加一维之后乘法就能搞定加法。逆矩阵就是 $T^{-1} = T(-t_x, -t_y)$，平移量取反即可。

---

### 二、旋转（Rotation）与等距变换（Isometry）

**纯旋转**（绕原点）：

$$R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta & 0 \ \sin\theta & \cos\theta & 0 \ 0 & 0 & 1 \end{bmatrix}$$

**等距变换**（旋转 + 平移，也叫刚体变换）：

$$M_{\text{iso}} = \begin{bmatrix} \cos\theta & -\sin\theta & t_x \ \sin\theta & \cos\theta & t_y \ 0 & 0 & 1 \end{bmatrix}$$

旋转矩阵有两个重要性质：$R^\top R = I$（正交矩阵），$\det(R) = 1$。这意味着 $R^{-1} = R^\top$——求逆只需转置，计算极其廉价。3D 中的旋转矩阵完全一样，只是升到 $3\times3$，并有绕 X、Y、Z 三轴的变体。

---

### 三、相似变换（Similarity）

旋转 + 平移 + **均匀缩放**，保持形状但允许大小变化：

$$M_{\text{sim}} = \begin{bmatrix} s\cos\theta & -s\sin\theta & t_x \ s\sin\theta & s\cos\theta & t_y \ 0 & 0 & 1 \end{bmatrix}$$

保持所有角度不变，但长度按比例 $s$ 缩放。人脸对齐（align faces to canonical size）就是用相似变换——把不同大小的脸归一化到同一尺寸。

---

### 四、仿射变换（Affine）

上半部分是任意 $2\times3$，底行固定为 $[0\ 0\ 1]$：

$$M_{\text{aff}} = \begin{bmatrix} a_{11} & a_{12} & t_x \ a_{21} & a_{22} & t_y \ 0 & 0 & 1 \end{bmatrix}$$

6 个自由度。最关键的不变量：**平行线变换后仍然平行**，面积比不变。三对匹配点就能唯一确定仿射变换（6 个方程，6 个未知数）。文档扫描纠斜、卫星图配准通常用仿射变换。

---

### 五、透视变换 / 单应矩阵（Homography）

底行不再是 $[0\ 0\ 1]$，这一行引入了真正的透视效果：

$$H = \begin{bmatrix} h_{11} & h_{12} & h_{13} \ h_{21} & h_{22} & h_{23} \ h_{31} & h_{32} & 1 \end{bmatrix}$$

8 个自由度（整体可乘以任意非零常数，故 9 个元素只有 8 个独立）。需要 4 对匹配点确定。输出的第三分量 $w = h_{31}x + h_{32}y + 1 \neq 1$，还原坐标需除以 $w$。

---

### 六、3D 投影矩阵（Camera Projection）

这是唯一一个真正把维度从 3D 降到 2D 的变换，矩阵形状是 $3\times4$：

$$\underbrace{\begin{bmatrix} f_x & 0 & c_x \ 0 & f_y & c_y \ 0 & 0 & 1 \end{bmatrix}}_{K\ \text{内参}} \underbrace{\begin{bmatrix} r_{11} & r_{12} & r_{13} & t_x \ r_{21} & r_{22} & r_{23} & t_y \ r_{31} & r_{32} & r_{33} & t_z \end{bmatrix}}_{[R|t]\ \text{外参}} \begin{bmatrix} X \ Y \ Z \ 1 \end{bmatrix} = \begin{bmatrix} wu \ wv \ w \end{bmatrix}$$

---


**关键原则**：模型的自由度越高，需要的匹配点越多，对噪声越敏感。在满足问题需求的前提下，**永远选自由度最低的模型**。用单应矩阵做只有旋转的拼接完全可以，但用仿射矩阵做有透视变形的拼接就会失败。


---

# RANSAC

**直觉理解：**
数据中混有大量错误点（outliers），如何在不知道哪些点是错误的情况下，依然估计出一个可靠的模型？
普通最小二乘对所有点一视同仁，一个严重偏离的点就能把结果拖垮。RANSAC 的思路完全不同——与其试图容忍异常点，不如随机找一个没有异常点的子集，在这个干净的子集上建模

**整体流程：**
![[Pasted image 20260325183420.png]]




### 关键问题：需要迭代多少次？

设内点占比为 $\varepsilon$（inlier ratio），每次采样 $s$ 个点。

单次采样 $s$ 个点**全部是内点**的概率：

$$P_{\text{good}} = \varepsilon^s$$

单次采样**至少一个外点**（模型可能错）的概率：

$$P_{\text{bad}} = 1 - \varepsilon^s$$

$N$ 次迭代**全都失败**的概率：

$$P_{\text{all fail}} = (1 - \varepsilon^s)^N$$

我们希望成功概率 $\geq p$（通常取 0.99），于是：

$$\boxed{N = \frac{\log(1-p)}{\log(1-\varepsilon^s)}}$$

感受内点比例和采样数量对迭代次数的影响：
可以看到两个关键规律：
内点比例从 50% 降到 30%，所需迭代次数可以增加 10 倍以上；最小采样数 $s$ 越大（比如单应矩阵需要 $s=4$，而直线只需要 $s=2$），收敛也更慢。

这就是为什么特征匹配质量（Lowe's ratio test）如此重要——它直接决定了内点比例，进而决定 RANSAC 需要跑多少轮。

### 三个关键参数的设置原则

$s$（最小采样数）是由模型本身决定的，没有调整空间——直线需要 2 点，单应矩阵需要 4 点，基础矩阵需要 7 点。真正需要调的是 $\varepsilon$ 和 $N$：

![[Pasted image 20260325184356.png]]


## RANSAC 的变体

标准 RANSAC 已经够用，但实际工程中有几个重要改进值得了解：

**PROSAC（Progressive Sample Consensus）**：不是完全随机采样，而是优先从置信度最高的匹配点中采样（比如描述子距离最小的前 K 对）。早期迭代有更高概率全部采到内点，收敛速度可以快 10× 以上。这是 OpenCV 中 `cv2.findHomography` 的 `USAC_PROSAC` 选项。

**MSAC（M-estimator Sample Consensus）**：对内点也按误差大小加权，不是简单计数内点数，而是最大化加权分。对边界附近的点更公平，精度略好于标准 RANSAC。

**LO-RANSAC（Locally Optimized）**：每次找到比当前最优更多的内点后，立刻用这些内点做一次局部最小二乘精化，再重新计算内点数。减少了最终精化步骤中模型跳变的风险。

**自适应 RANSAC（Adaptive / USAC）**：每轮结束后用当前最优内点率 $\hat\varepsilon$ 重新计算剩余所需迭代次数 $\hat N$，当剩余次数降到 0 时提前终止，平均迭代次数远小于固定 $N$。

---

## 完整流程在图像拼接中的位置

```
SIFT 检测 → 描述子匹配 → Lowe's ratio test → RANSAC → 内点精化
                                    ↑                    ↓
                           提高 ε（内点率）        最小二乘精化 H
                           RANSAC 收敛更快        内点越多，H 越准
```

Lowe's ratio test 把内点率从可能的 30–40% 提升到 60–80%，这使 RANSAC 所需迭代次数从数百次降到几十次。最后用全部内点（而非采样的 4 个点）对 $H$ 做一次最小二乘精化，这一步通常能把重投影误差再降低 30–50%。