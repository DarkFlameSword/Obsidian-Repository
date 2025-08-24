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
- 避免使用