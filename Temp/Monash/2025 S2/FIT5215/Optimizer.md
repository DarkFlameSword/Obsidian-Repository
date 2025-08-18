---
date: 2025-08-18
author:
  - Siyuan Liu
tags:
  - FIT5215
aliases:
  - summary
---
# SGD
$$𝑊 = 𝑊 − 𝜂 \frac{\partial{l}}{\partial{W}}$$
$𝜂$: learning rate
$l$: **单个小批量 (mini-batch) 数据** 的损失
$L$: 在 **整个训练数据集** 上计算出的**总损失 (Total Loss)** 或 **平均损失 (Average Loss)**
$$b = b − 𝜂 \frac{\partial{l}}{\partial{b}}$$
