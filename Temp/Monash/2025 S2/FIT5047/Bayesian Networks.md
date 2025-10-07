---
date: 2025-10-07
author:
  - Siyuan Liu
tags:
  - FIT5047
---
**定义：**
A data structure that represents the dependence between random variables

# Conditional Probability Tables (CPTs)

![[Pasted image 20251007215947.png]]

# D-separation
## Chain
```
X → M → Y
```
如果M被观测到，则X与Y独立
如果M没被观测到，则X与Y不独立

## Fork
```
X ← M → Y
```
如果M被观测到，则X与Y独立
如果M没被观测到，则X与Y不独立

## Collider
```
X → M ← Y
```
如果M被观测到，则X与Y不独立
如果M没被观测到，则X与Y独立

# Inference by Enumeration
**举例：**
```
    Cloudy (C)
       |
       ↓
    Rain (R)
       |
       ↓
  Wet Grass (W)
```

CPT:
**P(C)**

|Cloudy|P(C)|
|---|---|
|True|0.5|
|False|0.5|

**P(R | C)**

|Cloudy|Rain|P(R\|C)|
|---|---|---|
|True|True|0.8|
|True|False|0.2|
|False|True|0.2|
|False|False|0.8|

**P(W | R)**

|Rain|Wet|P(W\|R)|
|---|---|---|
|True|True|0.9|
|True|False|0.1|
|False|True|0.2|
|False|False|0.8|

**查询：P(C | W=True) = ?**
即：已知草地湿了，多云的概率是多少？

变量分类：
- 查询变量 (Q): C (Cloudy)
- 证据变量 (E): W = True
- 隐藏变量 (U): R (Rain) - 需要被求和消除

**需要计算两个值：**
$$\begin{aligned}
&P(C=True,W=True)\\
&P(C=False,W=True)\\
\end{aligned}$$
**P(C=True, W=True):**
$P(C=T,W=T)=\sum_rP(C=T)×P(r|C=T)×P(W=T|r)$

**R = True:**
$=P(C=T)×P(R=T|C=T)×P(W=T|R=T) =0.5×0.8×0.9 =0.36$

**R = False:**
$=P(C=T)×P(R=F|C=T)×P(W=T|R=F) =0.5×0.2×0.2 =0.02$

**求和：**
$P(C=T,W=T)=0.36+0.02=0.38$

**P(C=False, W=True):**'
$P(C=F,W=T)=\sum_rP(C=F)×P(r|C=F)×P(W=T|r)$

**R = True:**
$=P(C=F)×P(R=T|C=F)×P(W=T|R=T) =0.5×0.2×0.9 =0.09$

**R = False:**
$=P(C=F)×P(R=F|C=F)×P(W=T|R=F) =0.5×0.8×0.2 =0.08$

**求和：**
$P(C=F,W=T)=0.09+0.08=0.17$

**归一化:**
$P(C=T|W=T)=\frac{1}{P(W=T)}×0.38=1.818×0.38≈0.691$

$P(C=F|W=T)=\frac{1}{P(W=T)}×0.17=1.818×0.17≈0.309$