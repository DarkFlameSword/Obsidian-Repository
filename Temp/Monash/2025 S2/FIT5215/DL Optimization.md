---
date: 2025-08-24
author:
  - Siyuan Liu
tags:
  - FIT5215
aliases:
  - summary
---
# Gradient vanishing
**定义:**
Gradients get smaller and smaller as the algorithm progresses down to the lower layers

**多发场景:**
- activate function 使用的是`Sigmoid`,`tanh`

**解决办法:**
- 避免使用`Sigmoid`,`tanh`
- 选择合适的W,b初始化
    - `Xavier`/`Glorot` 初始化（适合 `tanh`/线性模型）
    - He 初始化（适合`ReLU`)
- 通过保持中间特征的分布稳定，避免梯度消失/爆炸
    - Batch Normalization (BN) —— 训练加速、梯度稳定
    - Layer Normalization (LN) —— 特别适合 RNN/Transformer
    - Group Normalization (GN)、Instance Norm 等
- 残差连接（Residual Connections）
    - ResNet 中的 shortcut/skip connection 可以让梯度绕过深层传播，极大缓解梯度消失
- 