---
date: 2025-10-05
author:
  - Siyuan Liu
tags:
  - FIT5047
---
**理解：**
朴素贝叶斯分类器基于两个核心假设：
1. **贝叶斯定理**：利用先验概率和条件概率进行推理
2. **特征独立性假设**（"朴素"的来源）：假设特征之间相互独立

**举例：**
```
目标：给定特征 X，预测类别 Y

核心思想：
P(Y|X) = P(X|Y) × P(Y) / P(X)

选择概率最大的类别：
Y* = argmax P(Y|X)
      Y
```

**数学公式：**
Assumes conditional independence of the attribute values for different classes
$$\begin{aligned}
c^* = &argmax_c P(c|v_{i1},v_{i2},\dots,v_{in})\\
&argmax_c \Pi_{k=1}^n P(v_{ik}|c)P(c)
\end{aligned}$$