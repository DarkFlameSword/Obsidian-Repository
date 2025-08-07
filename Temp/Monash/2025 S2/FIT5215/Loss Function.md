---
date: 2025-08-07
tags:
  - FIT5215
author:
  - Siyuan Liu
aliases:
  - base
---
# Classification
## `Kullback-Leibler (KL) divergence`
个人理解: KL散度用来计算, 当前模型距离真实案例的偏差值
$$
D_{KL}(P || Q) = \sum_{x} P(x) \cdot \log_2 \left( \frac{P(x)}{Q(x)} \right)
$$
==![[Pasted image 20250728164017.png]]==

![[Pasted image 20250727172836.png]]
## `Cross-entropy (CE)`
个人理解: 计算预测结果与真实结果之间的差距, CE越大预测结果越离谱

$$
H(p, q) = - \sum_{x} p(x) \cdot \log_2(q(x))
$$
==![[Pasted image 20250728164017.png]]

![[Pasted image 20250727174454.png]]
### `Entropy of the Distribution`
个人理解: 衡量的是一个随机事件或一个概率分布的**不确定性**或**混乱程度**, 也就是要搞清楚一个随机事件的最终结果，平均需要多少信息量（通常用比特`bit`来衡量）
- **熵越高**，代表系统越混乱，结果越不可预测
- **熵越低**，代表系统越有序，结果越容易预测

**H(X) = - Σ p(x) * log₂(p(x))**
==![[Pasted image 20250728164017.png]]

![[Pasted image 20250727173520.png]]

==Attention:==
1. CrossEntropy(p, q) = Entropy(p) + KL_Divergence(p || q) **在机器学习中, 真实数据分布 `p` 是固定的, 所以 `Entropy(p)` 是一个常数, 所以最小化交叉熵就等价于最小化KL散度**
2. CE越接近0, 则说明该事件越接近真实概率
# Regression

# `L1 Norm Loss / Mean Absolute Error Loss (MAE)`

$$ L = \frac{1}{n} \sum_{i=1}^{n} |y_i - p_i|$$
`y_i` 是真实值，`p_i` 是模型预测值，`n` 是样本数量
# `L2 Norm Loss / Mean Squared Error Loss(MSE)`

$$ L = \frac{1}{n} \sum_{i=1}^{n} |y_i - p_i|^2$$
`y_i` 是真实值，`p_i` 是模型预测值，`n` 是样本数量



