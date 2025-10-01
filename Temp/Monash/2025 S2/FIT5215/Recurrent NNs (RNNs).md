---
date: 2025-10-01
author:
  - Siyuan Liu
tags:
  - FIT5215
---
**核心特点：**
RNN 是一种专门处理**序列数据**的神经网络，其核心特点是：
**具有内部记忆（隐藏状态），可以处理任意长度的序列**
- **马尔可夫链**：离散状态转移，只看前一个状态
- **自回归模型**：线性模型，依赖过去 p 个观测值
- **RNN**：非线性模型，通过隐藏状态保存所有历史信息

# 简单RNNs理解（有输入无输出）
**计算过程：**
![[Pasted image 20251001144048.png]]

$$\begin{aligned}
& x_0 \in \mathbb{R}^{1 \times \textit{input\_size}}\\
& x_1 \in \mathbb{R}^{1 \times \textit{input\_size}}\\
& U \in \mathbb{R}^{\textit{input\_size} \times \textit{hidden\_size}}\\
& W \in \mathbb{R}^{\textit{hidden\_size} \times \textit{hidden\_size}}\\
\end{aligned}$$
$$h_0 = \tanh(x_0 U + b) \in \mathbb{R}^{1 \times \textit{hidden\_size}}$$
$$h_1 = \quad = \tanh(h_0 W + x_1 U + b) \in \mathbb{R}^{1 \times \textit{hidden\_size}}$$
# 有输入有输出
![[Pasted image 20251001150717.png]]
 $$\begin{aligned}
 & x_0, x_1, \ldots, x_t, \ldots \in \mathbb{R}^{1 \times \textit{input\_size}}\\
 & y_0, y_1, \ldots \in Y
 \end{aligned}$$
$$h_0 = \tanh(x_0 U + b)$$
$$\hat{y}_0 = \begin{cases} 
h_0 V + c & \text{(regression)} \\
\text{softmax}(h_0 V + c) & \text{(classification)}
\end{cases}$$
Suffer loss $l(\hat{y}_0, y_0)$

for $t = 1, 2, \ldots, T$
  $h_t = \quad = \tanh(h_{t-1} W + x_t U + b)$
  $$\hat{y}_t = \begin{cases}
  h_t V + c & \text{(regression)} \\
  \text{softmax}(h_t V + c) & \text{(classification)}
  \end{cases}$$
Suffer loss $l(\hat{y}_t, y_t)$

$\textit{Total\_loss} = \sum_{t=0}^{T} l(\hat{y}_t, y_t)$

# 多层RCNNs
![[Pasted image 20251001160155.png]]
# Forward propagation through the time
![[Pasted image 20251001160405.png]]
# Back propagation through the time
![[Pasted image 20251001160427.png]]

# RNNs模型类别
## One-to-One  (传统神经网络)
## One-to-Many     (图像描述生成)
![[Pasted image 20251001163754.png]]

---
## Many-to-One     (情感分析、分类)
![[Pasted image 20251001163747.png]]

---

## Many-to-Many    (机器翻译、视频分类) 
![[Pasted image 20251001163804.png]]
![[Pasted image 20251001163811.png]]

---
# RNNs核心问题
无法有效捕获长期依赖关系

**举例：**
![[Pasted image 20251001164544.png]]
**为什么能成功？**
- 预测 "sky" 所需的上下文（"the"）距离很近
- 隐藏状态 h 只需要传递几步就能保留足够的信息
- **短期依赖**：相关信息和需要该信息的位置距离很近

![[Pasted image 20251001164611.png]]
**为什么失败？**
- 预测 "Vietnamese" 需要的关键信息 "Vietnam" 距离太远
- 信息需要经过很多个隐藏状态传递
- 每次传递都会有**信息损失和衰减**
- **长期依赖**：相关信息和需要该信息的位置距离很远

