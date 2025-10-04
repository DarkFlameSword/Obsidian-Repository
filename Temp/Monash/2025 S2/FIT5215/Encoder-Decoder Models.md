---
date: 2025-10-03
author:
  - Siyuan Liu
tags:
  - FIT5215
---
**简单理解：**
```
输入序列        Encoder        上下文向量           Decoder        输出序列
   ↓              ↓                ↓                ↓               ↓
"How are you" → [编码] →     [0.2, -0.5, ...] →   [解码] →         "你好吗"
```
**Encoder:**
将输入序列编码成固定长度的**上下文向量**（Context Vector）

**Decoder:**
根据上下文向量生成目标序列

**数学公式：**
![[Pasted image 20251004150929.png]]We need to maximize the log-likelihood:
$$max\left(J(\theta) = \sum_{(x,y)\in D} log{P(y|x,(\theta_e, \theta_d)})\right)$$
![[Pasted image 20251004162538.png]]
在序列生成任务中，模型需要从大量可能的序列中选择最优的输出，所以需要用到[[Beam Search]]
