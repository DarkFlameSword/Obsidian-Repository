---
date: 2025-08-18
author:
  - Siyuan Liu
tags:
  - FIT5215
aliases:
  - summary
---
# SGD (Stochastic Gradient Descent)

**实际应用中的折中：小批量梯度下降 (Mini-batch GD)** 在实践中，我们通常不用纯粹的SGD（1个样本），也不用Batch GD（全部样本）。我们采用一个折中方案：每次随机取一小批数据（比如32、64或128个样本），用这个“mini-batch”来计算梯度并更新参数。这既利用了GPU并行计算的优势，又保持了更新的快速和随机性。现在人们通常说的SGD，大多指的都是这种Mini-batch SGD
$$𝑊 = 𝑊 − 𝜂 \frac{\partial{l}}{\partial{W}}$$
$𝜂$: learning rate
$l$: **单个小批量 (mini-batch) 数据** 的损失
$$b = b − 𝜂 \frac{\partial{l}}{\partial{b}}$$
