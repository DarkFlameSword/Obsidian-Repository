---
date: 2025-10-07
author:
  - Siyuan Liu
tags:
  - FIT5047
---
**术语概念：**
- Experiment：produces one of several possible outcomes
- Sample space： the set of all possible outcomes
- Event：a subset of the sample space
- Random variable：a variable whose value is determined by the outcome of an experiment
- Probability function：a function that assigns a probability to every possible outcome of an experiment

# Probabilistic Inference
## Conditional Probability
**术语概念：**
- Evidence: Agent knows certain things about the state of the world
- Hidden variables: Agent needs to reason about other aspects
- Model: Agent knows something about how the known variables relate to the unknown variables

---
# Kolmogorov’s axioms
==注意：==
该公理只针对有限离散随机变量

1. Non-negativity： 随机变量不能为负
2. Certainty：随机变量所有取值的和为1
3. Additivity：互斥事件的并集等于他们的和
---
# Probability Distributions
## Joint Distribution
**理解：**
描述两个或多个随机变量的联合行为
![[Pasted image 20251007201646.png]]

---
## Marginal Distribution
**理解：**
从联合分布中得到单个变量的分布
![[Pasted image 20251007201911.png]]

---
## Conditional Distribution
**术语概念：**
- \[Conditional / posterior\] probability: Pr(cavity | toothache)=0.8

**理解：**
描述在已知一个变量值的条件下，另一个变量的分布
![[Pasted image 20251007202058.png]]

Conditional Distribution 的计算公式：
$$P(X|Y) = \frac{P(X\land Y)}{P(Y)}$$

**Notation for conditional distributions:**
- Pr(cavity | toothache) = a single number
	- P(Cavity=True | Toothache=True) = 0.8
- Pr(Cavity, Toothache) = 2x2 table sums to 1

| Cavity \ Toothache | Toothache=T | Toothache=F | **行和**   |
| ------------------ | ----------- | ----------- | -------- |
| **Cavity=T**       | 0.12        | 0.08        | 0.20     |
| **Cavity=F**       | 0.08        | 0.72        | 0.80     |
| **Sum**            | 0.20        | 0.80        | **1.00** |
- Pr(Cavity | Toothache) = Two 2-element vectors, each sums to 1
	- 向量1：P(Cavity | Toothache=True): 
		- Cavity=True -> 0.60 
		- Cavity=False-> 0.40
	- 向量2：P(Cavity | Toothache=False):
		- Cavity=True -> 0.10 
		- Cavity=False-> 0.90

---
## Normalization Trick
如果你想通过Joint Distribution来得到某一个Random Variable在指定evidence下的Conditional Probability Distribution。可以通过挑选出指定evidence下的所有数据行，然后计算P的比例，这个比例就是P(Random Variable | evidence)的Conditional Probability Distribution

![[Pasted image 20251007204046.png]]

---
# Inference by Enumeration
**举例：**
查询：P(D | S=Present) = ?

即：已知患者有症状，患病的概率是多少？

变量分类：
- 查询变量 (Q): D (Disease)
- 证据变量 (E): S = Present
- 隐藏变量 (U): T (需要被"求和消除")

JPT：

|D|T|S|P(D)|P(T\|D)|P(S\|T)|**P(D,T,S)**|
|---|---|---|---|---|---|---|
|Yes|Pos|Pres|0.01|0.98|0.90|0.01×0.98×0.90 = **0.00882**|
|Yes|Pos|Abs|0.01|0.98|0.10|0.01×0.98×0.10 = **0.00098**|
|Yes|Neg|Pres|0.01|0.02|0.20|0.01×0.02×0.20 = **0.00004**|
|Yes|Neg|Abs|0.01|0.02|0.80|0.01×0.02×0.80 = **0.00016**|
|No|Pos|Pres|0.99|0.05|0.90|0.99×0.05×0.90 = **0.04455**|
|No|Pos|Abs|0.99|0.05|0.10|0.99×0.05×0.10 = **0.00495**|
|No|Neg|Pres|0.99|0.95|0.20|0.99×0.95×0.20 = **0.18810**|
|No|Neg|Abs|0.99|0.95|0.80|0.99×0.95×0.80 = **0.75240**|

**Step1: Select entries consistent with the evidence**
只保留 S=Present

| D   | T   | S    | P(D,T,S)    |
| --- | --- | ---- | ----------- |
| Yes | Pos | Pres | **0.00882** |
| Yes | Neg | Pres | **0.00004** |
| No  | Pos | Pres | **0.04455** |
| No  | Neg | Pres | **0.18810** |
**Step2: Sum out U to get a joint probability of Q and E**
T已被消除

| D   | S    | P(D, S=Pres) |
| --- | ---- | ------------ |
| Yes | Pres | **0.00886**  |
| No  | Pres | **0.23265**  |

**Step3: Normalize the remaining entries to conditionalize**

|D|P(D \| S=Pres)|
|---|---|
|**Yes**|**0.0367** (3.67%)|
|**No**|**0.9633** (96.33%)|

---
# Product Rule
将conditional distribution 转化为 joint distribution

$$P(X,Y) = P(X|Y)P(Y) $$
---
# Chain Rule
对于n个随机变量 X₁, X₂, ..., Xₙ，它们的联合概率可以分解为：
$$P(X₁, X₂, ..., Xₙ) = P(X₁) \;×\; P(X₂|X₁) \;×\; P(X₃|X₁,X₂) \;×\; ... \;×\; P(Xₙ|X₁,...,Xₙ₋₁)$$
==特殊情况：==
```
    C (Cloudy)
    |
    ↓
    R (Rain)
    |
    ↓
    W (Wet Grass)
```
在已知 R 的情况下，W 与 C 条件独立
$P(W|R,C)=P(W|R)$
$$P(C,r,W=True)=P(C)×P(r|C)×P(W=True|r)$$
# Bayes Rule
$$
P(X|Y) = \frac{P(Y|X)P(X)}{P(Y)}
$$
$$
\text{Posterior Probability} = \frac{\text{Likelihood}\times \text{Prior Probability}}{\text{Probability of Evidence}}
$$

# Independence
Two variables are independent if:
$$P(X,Y) = P(X)P(Y)$$
记作：
$X \perp Y$

## Conditional Independence
Two variables are independent if:
$$P(X,Y|Z) = P(X|Z)P(Y|Z)$$
