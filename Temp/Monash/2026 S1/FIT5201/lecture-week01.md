---
date: 2026-03-06
author:
  - Siyuan Liu
tags:
  - FIT5201
---
# Machine Learning

定义：
如果一个程序在某类任务 T 上、以某种经验 E 为依据，其在评价指标 P 上的性能随着经验 E 的积累而自动提高，那么我们就说该程序从经验 E 中进行了学习

### Model
**Parametric model:**
parameters are fixed regardless of the size of the training set (e.g. Linear regression)

**Non-parametric model:**
the number of parameters can grow as the size of training set increases (e.g. k-NN classifier)

**Linear Model:**
模型输出与输入特征之间，在某个特征空间里是“线性关系”

常见线性模型：
- 线性回归（Linear Regression）
- 岭回归 / Lasso（本质仍是线性模型，只是正则化不同）
- 逻辑回归（Logistic Regression：在特征空间是线性决策边界）
- 线性 SVM（Linear SVM）

**Non-Linear Model:**
模型输出与输入特征之间，在任何特征空间里是“非线性关系”

注意：
对于$y(w,x) = w_1 x + w_2x^2 + w_3x^3\dots w_Mx^M$ 注意到对于$w$的$y$模型是线性的，但是$w$和$x$的联合$y$模型来讲任然是线性的

### MSE（Mean Squared Error Function）
$$E(w) = \frac{1}{N} \sum_{n=1}^N [y(x_n,w)-t_n]^2$$
- $y(x_n,w):$ 预测值
- $t_n:$ 真实标签值
- $N:$ 数据集大小

所以我们的优化目标是：$w^* = argmin_w E(w)$
同梯度来说就是：$\frac{\partial(E(w))}{\partial(w)} = 0$

### How to measure the generalization performance on a model with M?
RMS(Root Mean Square): $E_{RMS} = \sqrt(\frac{2E(w*)}{N}) \to \text{N is the size of dataset}$ 

# Regularization

有很多技巧都是对数据进行正则化，在Error Function后面加惩罚项是常见的办法之一
$$
\begin{align}
& E(w) = \frac{1}{2} \sum_{n=1}^N [y(x_n,w)-t_n]^2+P(w)\\
& P(w) = \frac{\lambda}{2}||w||^2
\end{align}
$$

| **λ 取值**            | **模型复杂度** | **对w的影响** |
| ------------------- | --------- | --------- |
| **增加 $\uparrow$**   | 降低（更僵化）   | 惩罚大w      |
| **适中**              | 适中        | Good Fit  |
| **减少 $\downarrow$** | 增加（更灵活）   | 惩罚小w      |

---

# K-Fold Cross-Validation K 折交叉验证
### 场景设定

你有一个很小的数据集，一共 10 个样本，要训练一个房价预测模型（比如线性回归），但你又想比较可靠地估计：
- 这个模型在“没见过的新数据”上，大概能有多好？

如果直接“8 个样本训练 + 2 个样本做验证”，有两个问题：
1. 验证集太小，评估结果很不稳定（运气成分大）
2. 具体拿哪 2 个做验证，也很主观

K 折交叉验证的目的就是：**让每一个样本都轮流当一次“验证集”，然后取多次评估结果的MSE的均值，使得评估更稳定、更公平

### 什么是“K 折”（拿 K=5 做例子）

假设 K = 5，把 10 个样本平均分成 5 份（每份 2 个样本）：
- Fold 1：样本 1, 2
- Fold 2：样本 3, 4
- Fold 3：样本 5, 6
- Fold 4：样本 7, 8
- Fold 5：样本 9, 10
（真实情况一般是打乱顺序后再分）
然后做 5 次“训练 + 验证”：

**第 1 折：**
- 训练集：Fold 2 + Fold 3 + Fold 4 + Fold 5（样本 3–10） → 共 8 个样本
- 验证集：Fold 1（样本 1–2）
- 得到一个模型，算出在样本 1–2 上的误差，比如 MSE₁

**第 2 折：**
- 训练集：Fold 1 + Fold 3 + Fold 4 + Fold 5（样本 1–2, 5–10）
- 验证集：Fold 2（样本 3–4）
- 得到第二个模型，算出在样本 3–4 上的误差 MSE₂

**第 3 折：**
- 训练集：Fold 1 + Fold 2 + Fold 4 + Fold 5
- 验证集：Fold 3
- 得到第三个模型，误差 MSE₃

**第 4 折：**
- 训练集：Fold 1 + Fold 2 + Fold 3 + Fold 5
- 验证集：Fold 4
- 得到第四个模型，误差 MSE₄

**第 5 折：**
- 训练集：Fold 1 + Fold 2 + Fold 3 + Fold 4
- 验证集：Fold 5
- 得到第五个模型，误差 MSE₅
## 3. 得到总体评估指标

现在有 5 个“在未见数据上的”误差：

$MSE_1,MSE_2,MSE_3,MSE_4,MSE_5$

K 折交叉验证的最终评估通常取**平均值**：

$\text{MSE}_{\text{CV}} = \frac{1}{5} \sum_{i=1}^{5} \text{MSE}_i$
- 这个平均误差就被当作“模型在新数据上的预期表现”
- 有时也会看一下**标准差**，来衡量模型性能是否稳定

---

# Leave-One-Out Cross-Validation

其实就是“K 折交叉验证里，的极端情况”也就是：**每次只留 1 个样本做验证，剩下所有样本都拿来训练**
