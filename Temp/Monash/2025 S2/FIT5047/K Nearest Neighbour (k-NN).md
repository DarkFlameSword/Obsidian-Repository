---
date: 2025-10-05
author:
  - Siyuan Liu
tags:
  - FIT5047
---
**核心思想:**
**KNN** 是一种基于实例的学习算法（Instance-based Learning），也称为懒惰学习（Lazy Learning）

给定一个测试样本，找到与它最相似的 K 个训练样本，通过这 K 个"邻居"的类别来预测测试样本的类别

物以类聚：
- 你的邻居大多是医生 → 你可能也是医生
- 你周围都是程序员 → 你可能也是程序员

