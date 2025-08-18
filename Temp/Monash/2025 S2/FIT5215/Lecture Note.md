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
- **目标**：学习一个函数 `f`，使得对于新的、从未见过的数据 `X_new`，模型能够预测出正确的标签 `y_new`，即 `y_new ≈ f(X_new)`。
### Classification
**核心思想：** 预测一个**离散的类别标签**。简单说，就是做“选择题”。

模型的任务是将输入数据分到一个预先定义好的类别中。输出的结果是有限的、不连续的类别。

**目标：** 输出是一个**类别 (Class)**。
#### 计算Loss的公式
1. **Mean Squared Error**
```
MSE = (1/n) * Σ(yᵢ - ŷᵢ)²
```
- `n`: 样本总数
- `yᵢ`: 第 `i` 个样本的真实值
- `ŷᵢ`: 第 `i` 个样本的预测值

==Attention:==
- 将这个误差**求平方**。这使得所有误差都为正数，并且会**极大地惩罚那些误差很大的预测**（比如真实值是10，你预测成100，平方后差距会变得非常大）

2. **Mean Absolute Error**
```
MAE = (1/n) * Σ|yᵢ - ŷᵢ|
```
==Attention:==
-  MAE 对所有大小的误差都给予相同的线性惩罚。它对**异常值 (Outliers)** 不像 MSE 那么敏感。如果你的数据中有很多离谱的异常点，使用 MAE 可能会让模型更稳定
### Regression
**核心思想：** 预测一个**连续的数值**。简单说，就是做“填空题”，填一个具体的数字。

模型的任务是基于输入数据，预测一个精确的、连续的输出值。

**目标：** 输出是一个**数值 (Value)**

#### 计算Loss的公式
1. **Cross-Entropy Loss**
```
BCE = - (1/n) * Σ [ yᵢ * log(ŷᵢ) + (1 - yᵢ) * log(1 - ŷᵢ) ]
```
- `yᵢ`: 真实标签，只能是 **0 或 1**。
- `ŷᵢ`: 模型预测的概率，通常是经过 `Sigmoid` 函数输出的，值在 (0, 1) 之间，表示样本为类别 1 的概率。

==Attention:==
- 如果真实标签 `y=1`: 公式简化为 `-log(ŷ)`。为了让 Loss 变小，`log(ŷ)` 就要变大，这意味着 `ŷ` 必须**趋近于 1
- 如果真实标签 `y=0`: 公式简化为 `-log(1 - ŷ)`。为了让 Loss 变小，`log(1 - ŷ)` 就要变大，这意味着 `(1 - ŷ)` 必须趋近于 1，也就是 `ŷ` 必须**趋近于 0**

1. **Categorical Cross-Entropy**
```
CCE = - (1/n) * Σ Σ [ yᵢ,c * log(ŷᵢ,c) ]
```
- `yᵢ,c`: 是一个 **One-Hot 编码**的向量。如果第 `i` 个样本的真实类别是 `c`，则 `yᵢ,c=1`，否则为 0。
- `ŷᵢ,c`: 模型预测的概率分布，通常是经过 `Softmax` 函数输出的，表示第 `i` 个样本属于类别 `c` 的概率。

==Attention:==
- 由于 One-Hot 编码的存在，对于每个样本 `i`，只有一个 `yᵢ,c` 是 1，其他都是 0。所以内层求和 `Σ` 会被简化
## Reinforce Learning

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
# SOTA Deep Neural Nets

## Multilayered feedforward neural nets
For working with vectors, 1D tensors
## Convolutional neural nets / ViT
For working with images, 2D/3D tensors
## Recurrent neural nets / Transformers
For working with sequences, sentences, texts

# Types of Games
1. Perfect information (deterministic, full observe.)
    Chess, Go, noughts-and-crosses (tic-tac-toe), draught (checkers), etc.
2. Imperfect information (stochastic, partially observe.)
    Poker, Texas hold’em (variant of poker), bridge, backgammon, etc
3. n-Players (e.g., n = 2 for chess)
4. Sequential (vs. simultaneous)
5. Zero-sum (vs non-zero sum games)

# Game Trees(博弈树)
## Concept
- **MAX**：希望最大化自己的得分
- **MIN**：希望最小化对手的得分
- A position favorable to MAX → utility > 0
    - 如果某个局面对 MAX 有利，就给它一个 **正数值**，越大越好。
    - MAX 获胜的终局，通常赋值 **+∞**
- A position favorable to MIN → utility < 0
    - 如果某个局面对 MIN 有利，就给它一个 **负数值**，越小越好
    - MIN 获胜的终局，通常赋值 **−∞**
- The set of nodes generated for one player is called a **ply**
    - **ply** 表示一层搜索深度：
    - 1 ply = 一个玩家下的一步棋   
    - 2 ply = 双方各下一步（即一个来回）
    - 3 ply = MAX → MIN → MAX
    ### Each node has an associated utility

- 每个状态（节点）都有一个 **效用值 (utility)**，用来评估该局面对双方的好坏。
