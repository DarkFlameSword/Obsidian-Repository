---
date: 2025-09-07
author:
  - Siyuan Liu
tags:
  - FIT5215
---
# Machine Learning Revision
1. 为什么要使用vector
		vector相较于以前人工提取feature, 能够被更好地被电脑处理
2. [[Mathematics#^42723b|vector]]
3. [[Mathematics#^839d47|p-normalization]]
4. [[Mathematics#^710368|Euclidean distance & cosine similarity & cosine distance]]
5. the proof of **Cosine distance can be computed via Euclidean distance if vectors are made unit vectors**
6. KL,CE,E的计算
7. ground_truth_label的概念
8. One_hot_label的概念
9. FFNNs的计算
10. 激发函数计算:ReLU,Tanh,sigmoid
11. Assume that we have 4 classes in {cat = 1,dog = 2,lion = 3, monkey = 4}. Given a data example 𝑥 with ground-truth label “dog”, assume that a feed-forward NN gives discriminative scores to this 𝑥 as ℎ1 = −3, ℎ2 = 10, ℎ3 = 5, ℎ4 = 0. What is the CE loss suffered by this prediction?就是求模型的预测值的CE
12. 