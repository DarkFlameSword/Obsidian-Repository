---
date: 2025-10-05
author:
  - Siyuan Liu
tags:
  - FIT5047
---
**核心思想:**
**KNN** 是一种基于实例的学习算法（Instance-based Learning），也称为懒惰学习（Lazy Learning）, KNN 算法本身没有“训练”过程，它不会像其他算法（如线性回归、决策树）那样从训练数据中学习一个判别函数或模型。相反，它只是将所有训练数据存储起来，直到有新的数据需要预测时，才开始进行计算

给定一个测试样本，找到与它最相似的 K 个训练样本，通过这 K 个"邻居"的类别来预测测试样本的类别

物以类聚：
- 你的邻居大多是医生 → 你可能也是医生
- 你周围都是程序员 → 你可能也是程序员

**计算步骤:**

1. **确定K值**：首先，需要选择一个整数K。K代表我们要参考的“邻居”的数量。K值的选择对最终结果有很大影响
2. **计算距离**：当有一个新的、未标记的数据点需要预测时，KNN会计算这个新点与训练数据集中**每一个**点之间的距离。
		离散距离度量是**欧几里得距离（Euclidean Distance）**，其公式为：
    $$
    d(p,q)=\sqrt{(p_1-q_1)^2 + (p_2-q_2)^2 + \dots + (p_n-q_n)^2}
    $$
		连续距离度量是**Jaccard coefficient**，其公式为：
		$$d(p,q)=\frac{[p_1,p_2,\dots]\cap [q_1,q_2,\dots]}{[p_1,p_2,\dots]\cup [q_1,q_2,\dots]}$$
- p 和 q 是两个数据点，它们分别有 n 个特征（维度）

1. **找到最近的K个邻居**：根据计算出的距离，找出离新数据点最近的K个训练样本
2. **投票/平均**：
    - **对于分类任务**：在这K个邻居中，统计每个类别出现的次数。出现次数最多的那个类别，就是新数据点的预测类别
    - **对于回归任务**：不是投票，而是计算这K个邻居的数值（例如，房价、温度等）的**平均值**或**加权平均值**，作为新数据点的预测值

![[Pasted image 20251015192204.png]]


Note that if the question had been to calculate the Jaccard coefficient to find nearest neighbours in kNN, we would not include the predicted value in the calculation