---
date: 2025-08-18
author:
  - Siyuan Liu
tags:
  - FIT5215
aliases:
  - summary
---
# Gradient Decent
$$\hat{y} = \sigma(w^T x + b), \quad \sigma(z) = \frac{1}{1 + e^{-z}}$$
- $\sigma(z)$: 使用`Sigmoid`对$z$激活
$$J(w, b) = \frac{1}{m} \sum_{i=1}^{m} \mathcal{L}(\hat{y}^{(i)}, y^{(i)}) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$
- `J(w, b)`: 关于参数 `w` 和 `b` 的成本函数
- $y^{(i)}$: 第 `i` 个训练样本的真实标签

