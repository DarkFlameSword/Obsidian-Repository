---
date: 2025-07-27
tags:
  - FIT5215
author:
  - Siyuan Liu
aliases:
  - note
---
# AI Model Types

## Supervised Learning
**核心思想：** 通过**有答案的“练习题”**来学习。

把它想象成一个学生（机器学习模型）在学习。我们为他提供大量的练习题（输入数据 `X`），并且**每一道题都附带了标准答案**（输出标签 `y`）。学生的目标就是学习从题目到答案之间的规律和映射关系。

**关键特征：**

- **使用“已标记”的数据 (Labeled Data)**：数据集中的每个样本都包含两部分：**特征 (Features)** 和与之对应的 **标签 (Label)** 或 **目标 (Target)**。
- **目标**：学习一个函数 `f`，使得对于新的、从未见过的数据 `X_new`，模型能够预测出正确的标签 `y_new`，即 `y_new ≈ f(X_new)`
### Classification

**核心思想：** 
预测一个**离散的类别标签**。简单说，就是做“选择题”

**特征:**
模型的任务是将输入数据分到一个预先定义好的类别中。输出的结果是有限的、不连续的类别。

**计算Loss的公式:**
1. **Mean Squared Error**

$$MSE = \frac{1}{n} * \sum{(p_i - q_i)^2}$$

- $n$: 样本总数
- $p_i$: 第 `i` 个样本的真实值
- $q_i$: 第 `i` 个样本的预测值

==Attention:==
- 将这个误差**求平方**。这使得所有误差都为正数，并且会**极大地惩罚那些误差很大的预测**（比如真实值是10，你预测成100，平方后差距会变得非常大）

2. **Mean Absolute Error**
$$MSE = \frac{1}{n} * \sum{(p_i - q_i)}$$
==Attention:==
-  MAE 对所有大小的误差都给予相同的线性惩罚。它对**异常值 (Outliers)** 不像 MSE 那么敏感。如果你的数据中有很多离谱的异常点，使用 MAE 可能会让模型更稳定
### Regression

**核心思想：** 
预测一个**连续的数值**。简单说，就是做“填空题”，填一个具体的数字

**特征:**
模型的任务是基于输入数据，预测一个精确的、连续的输出值

## Unsupervised Learning

**核心思想:**
任务不是去“预测”什么，而是去“发现”数据本身内在的结构、模式或关系

无监督学习主要解决两类问题：**聚类 (Clustering)** 和 **降维 (Dimensionality Reduction)**
### Clustering
**核心思想:**
将数据分成不同的组（簇），使得同一组内的数据点彼此相似，而不同组之间的数据点差异较大

#### K-Means
#### Hierarchical Clustering

### Dimensionality Reduction

**核心思想:**
在保留数据最重要信息的前提下，减少数据的特征数量（维度）

#### PCA

#### Autoencoders

#### t-SNE
---
## Reinforce Learning
**核心思想:**
通过与环境互动，不断试错，并根据得到的结果（奖励或惩罚）来调整自己的行为，最终学会如何做出最优的决策

**The learning Loop:**
1. 智能体观察当前**状态 (State)**
2. 智能体根据当前状态，选择一个**动作 (Action)** 并执行
3. 环境接收到动作后，会转变为一个新的**状态 (State)**
4. 同时，环境会给智能体一个**奖励 (Reward)**。
5. 智能体接收到新的状态和奖励，然后重复第一步

### Q-Learning

### Deep Q-Network (DQN)

# Data Set
##  Training Set: 80%
- 模型能直接看到、学习到的数据。
## Validation Set: 10
- 模型不能用它来更新参数，只是用来“考察”模型在训练集的效果。
## Test Set: 10%
- **特征**：完全独立于训练集和验证集，直到最后才用。

# Performance Metrics

### Accuracy
### Recall
### Precision
### F-score

# Deep Learning
## Where DL Works?
![[Pasted image 20250808104946.png]]
## The Relationship  between DL and ML
![[Pasted image 20250808105229.png]]
# Recall optimization problem in deep learning
### 1. 核心目标：最小化损失函数

想象一下，你正在一个浓雾弥漫的、连绵不绝的山脉中，你的目标是**走到山谷的最低点**。

- **你的当前位置**：代表了模型使用当前参数所达到的性能。
- **山脉的地形**：由一个叫做**损失函数 (Loss Function)**决定。这个函数衡量了模型的**预测结果**与**真实标签**之间的差距。差距越大，损失函数的值就越高（海拔越高）。
- **山谷的最低点**：代表了模型的**最优参数状态**，此时模型的预测误差最小。

所以，深度学习的优化问题，本质上就是一个**数学优化问题**：寻找一组能让损失函数 `L` 达到最小值的参数 `θ`。


$$\theta^*=arg⁡min_\theta L(\theta)$$

其中：

- `θ` (theta) 代表了模型中所有需要学习的参数（主要是权重 `W` 和偏置 `b`）。一个现代模型可能有数十亿个参数。
- `L(θ)` 是在整个训练数据集上计算出的总损失。
- `θ*` (theta-star) 是我们想要找到的那组最优参数。
### 2. 优化的三个核心组件

要定义这个优化问题，我们需要三个东西：

1. **参数 (Parameters, `θ`)**： 这些是模型内部可以调节的“旋钮”。我们的目标就是找到这些旋钮的最佳设置。
    
2. **目标函数 (Objective Function / Loss Function, `L`)**： 这是我们评估模型好坏的“海拔地图”。它必须是**可微的 (differentiable)**，这样我们才能计算梯度。常见的损失函数有：
    
    - **均方误差 (MSE)**：用于回归任务。
    - **交叉熵损失 (Cross-Entropy Loss)**：用于分类任务。
3. **数据 (Data)**： 损失函数是基于模型的预测和**训练数据**的真实标签来计算的。数据定义了我们正在优化的“山脉”的具体形状。
### 3. 求解方法：梯度下降及其变体

我们如何在浓雾中找到下山的路？最直观的方法是：**环顾四周，找到最陡峭的下坡方向，然后朝着那个方向迈一小步。** 不断重复这个过程，最终就能到达一个低点。

这正是**梯度下降 (Gradient Descent)** 算法的精髓。

#### a. 梯度 (Gradient, `∇L`)

- **什么是梯度？** 梯度是一个向量，它指向函数值**增长最快**的方向。在我们的比喻中，它指向**上山最陡**的方向。
- **如何使用？** 我们要下山，所以我们应该朝着**梯度的反方向 (`-∇L`)** 前进。这个方向是当前位置下降最快的方向。

#### b. 学习率 (Learning Rate, `α`)

- **是什么？** 学习率决定了我们沿着下坡方向**迈出的步子有多大**。
- **为什么重要？**
    - **学习率太大**：可能会“一步迈过”山谷的最低点，导致在谷底两侧来回震荡，甚至可能越走越高（发散）。
    - **学习率太小**：下山速度会非常慢，需要极长的时间才能收敛。

#### c. 核心更新规则

梯度下降的每一步都遵循这个简单的规则来更新模型的每一个参数：

**`新参数 = 老参数 - 学习率 × 梯度`** $$ \theta_{new} = \theta_{old} - \alpha \nabla L(\theta_{old}) $$

#### d. 挑战与现代优化器

简单的梯度下降在实践中面临很多挑战：

1. **计算成本高**：在大型数据集上，计算一次完整的梯度（遍历所有数据）非常耗时。
    
    - **解决方案**：**随机梯度下降 (SGD)** 及其变体。不是看遍整个山脉再决定走哪，而是在一小块区域（一个 **mini-batch** 的数据）上估计一个大致的下山方向，然后快速前进一步。虽然方向不那么准（很“抖动”），但更新速度快得多，总体效率更高。
2. **复杂的“地形”**：损失函数的“山脉”并非一个完美光滑的碗。它充满了：
    
    - **局部最小值 (Local Minima)**：我们可能走到了一个小山谷，但它不是全局最低点。
        
    - **鞍点 (Saddle Points)**：在某个方向上是最高点，在另一个方向上是最低点，梯度在此处为零，可能导致优化停滞。
        
    - **平原 (Plateaus)**：大片梯度几乎为零的平坦区域，优化进程会变得极其缓慢。
        
    - **解决方案**：**现代优化算法**，如 **Adam**, **RMSprop**, **Adagrad**。这些算法不仅仅看当前的梯度，还引入了**动量 (Momentum)** 和**自适应学习率 (Adaptive Learning Rate)** 的概念。
        
        - **动量**：像一个从山上滚下来的重球，即使遇到小的颠簸或平坦区域，也能凭借惯性冲过去。
        - **自适应学习率**：在平坦的区域迈大步，在陡峭、狭窄的区域迈小步，让优化过程更智能、更稳定。

# Gradient 
$${\nabla L(\theta)} = 
{
\begin{bmatrix} 
\frac{\partial}{\partial{\theta_1}} \\
\frac{\partial}{\partial{\theta_2}} \\
\frac{\partial}{\partial{\theta_3}} \\
\;\cdots\;\\
\frac{\partial}{\partial{\theta_n}}
\end{bmatrix}
}$$
对于一个输入为多维向量 `θ`（代表所有模型参数），输出为一个标量（损失值 `L`）的函数 `L(θ)`，梯度 `∇L(θ)` 是一个**向量**。这个向量的每个分量是损失函数 `L` 对每个参数 `θᵢ` 的**偏导数**

==也就是说:==
- 梯度向量 `∇L(θ)` 指向在当前参数点 `θ` 处，**损失函数 `L` 增长最快的方向**
- 在深度学习中，我们的目标是**最小化**损失函数。因此，我们不应该沿着梯度方向走，而应该沿着**梯度的反方向 (`-∇L`)** 走，因为这是损失函数**下降最快**的方向
- 这正是**梯度下降 (Gradient Descent)** 算法的核心思想： $$ \theta_{new} = \theta_{old} - \alpha \nabla L(\theta_{old}) $$ 其中 `α` 是学习率。梯度是驱动所有现代深度学习模型训练的**基本引擎**。它是**一阶优化算法**（First-Order Optimization）的基石
# Hessian Matrix
$$
H=
{\begin{bmatrix}
\frac{\partial L}{\partial \theta_1\theta_1} & \cdots & 
\frac{\partial L}{\partial \theta_1\theta_5} & \cdots & 
\frac{\partial L}{\partial \theta_1\theta_n} \\ \vdots & \ddots & \vdots & \ddots & \vdots \\
\frac{\partial L}{\partial \theta_5\theta_1} & \cdots & 
\frac{\partial L}{\partial \theta_5\theta_5} & \cdots & 
\frac{\partial L}{\partial \theta_5\theta_n} \\ \vdots & \ddots & \vdots & \ddots & \vdots \\
\frac{\partial L}{\partial \theta_n\theta_1} & \cdots & 
\frac{\partial L}{\partial \theta_n\theta_5} & \cdots & 
\frac{\partial L}{\partial \theta_n\theta_n}
\end{bmatrix}
}
$$
$$H(L) = ∇(∇L)$$
海森矩阵 `H` 是一个**方阵**，它包含了损失函数 `L` 所有的**二阶偏导数**。如果模型有 `n` 个参数，海森矩阵就是一个 `n x n` 的矩阵。
矩阵中第 `i` 行、第 `j` 列的元素是 `L` 先对 `θᵢ` 求导，再对 `θⱼ` 求导的结果。

==也就是说:==
海森矩阵描述了损失函数在某个点附近的**局部曲率 (local curvature)**。换句话说，它描述了“山谷”的几何形状。

- 如果海森矩阵在某点是**正定的 (positive-definite)**，意味着损失函数在该点附近是**向上凸的**（像一个碗），这表明我们处在一个**局部最小值 (local minimum)**。
- 如果海森矩阵是**负定的 (negative-definite)**，意味着损失函数是**向下凹的**（像一座山峰），这表明我们处在一个**局部最大值 (local maximum)**。
- 如果海森矩阵是**不定的 (indefinite)**（特征值有正有负），这表明我们处在一个**鞍点 (saddle point)**
- 海森矩阵是**二阶优化算法**（Second-Order Optimization），如**牛顿法 (Newton's Method)** 的核心。牛顿法的参数更新规则是： $$ \theta_{new} = \theta_{old} - H^{-1} \nabla L(\theta_{old}) $$ 这里 `H⁻¹` 是海森矩阵的逆。与梯度下降相比，牛顿法不仅考虑了下降最快的方向（梯度），还考虑了曲率（海森矩阵），从而能够更智能、更直接地跳向最小值点
# Local Minima, Maxima and Saddle Point
## Local Minima
![[Pasted image 20250818163944.png]]
- **Analogy:** The bottom of a valley or a bowl.
- **Description:** This is a point where the loss is lower than at all its immediate neighbors. It's a "good" place for the optimization to stop, as we have successfully minimized the loss in this local region.
- **Mathematical Conditions:**
    1. **Gradient is zero:** `∇L(θ) = 0`.
    2. **Curvature is positive in all directions:** The Hessian matrix `H` is **positive-definite**. This means if you move away from this point in any direction, the loss will increase. (All eigenvalues of the Hessian are positive).
## Local Maxima
![[Pasted image 20250818164006.png]]
- **Analogy:** The peak of a hill.
- **Description:** This is a point where the loss is higher than at all its immediate neighbors. It's a "bad" place to get stuck, but fortunately, it's very unstable.
- **Mathematical Conditions:**
    1. **Gradient is zero:** `∇L(θ) = 0`.
    2. **Curvature is negative in all directions:** The Hessian matrix `H` is **negative-definite**. This means if you move away from this point in any direction, the loss will decrease. (All eigenvalues of the Hessian are negative).
## Local Saddle Point
![[Pasted image 20250818164013.png]]
- **Analogy:** The middle of a mountain pass or a horse's saddle.
- **Description:** This is the most interesting and challenging type of critical point. From a saddle point, the loss goes **up** in some directions and **down** in others. It's a minimum along one axis but a maximum along another.
- **Mathematical Conditions:**
    1. **Gradient is zero:** `∇L(θ) = 0`.
    2. **Curvature is mixed:** The Hessian matrix `H` is **indefinite**. It has both positive and negative eigenvalues.


# Optimization Problem in ML and DL
Most of optimization problems in machine learning (deep learning) has the
following formula:
$$\min_{\theta}{J(\theta)} = \Omega(\theta) + \frac{1}{N}\sum_{i=1}^N l(y_i,f(x_i;\theta))$$


