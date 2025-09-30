---
date: 2025-09-01
author:
  - Siyuan Liu
tags:
  - FIT5215
aliases:
  - summary
---
# 前言
ResNet其实就是由Residual Block构成的神经网络，解决了深度网络梯度消失，无法继续学习的问题
# Residual Block
![[Pasted image 20250930203936.png]]
Residual Block的输出会加上输入，从而一定程度上保证模型学习的进度是在之前学习进度的基础上进行的，而不是越来越小直至消失
## Skip Filter
因为矩阵相加必须满足矩阵shape一致，所以假如Residual Block的输出与输入大小不一致，则需要Skip Filter去reshape输入