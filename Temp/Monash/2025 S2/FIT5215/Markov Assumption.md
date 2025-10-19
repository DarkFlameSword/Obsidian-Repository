---
date: 2025-10-01
author:
  - Siyuan Liu
tags:
  - FIT5215
---
# Markov Assumption
泛指马尔可夫性质：序列的未来状态在给定某些过去状态条件下只依赖这些过去状态，而与更早的历史无关

# First-order Markov Assumption
一类基于**马尔可夫假设**的概率模型，特指 k=1 的情形：未来只依赖于前一个状态, 用于描述系统在不同状态之间的随机转移过程

**数学公式：**
$$P(X_t+1 | X_t, X_t-1, X_t-2, ..., X_1) = P(X_t+1 | X_t)$$

**举例：**
今天的天气取决于昨天的天气 但与前天、大前天的天气无关 如果今天是晴天，明天可能： - 60% 晴天 - 30% 多云 - 10% 雨天 这个概率与前几天是什么天气无关！

# Markov Chain
**概念单元**
1. **状态空间 (State Space)**：所有可能状态的集合
2. **转移概率 (Transition Probability)**：从一个状态转移到另一个状态的概率
3. **转移矩阵 (Transition Matrix)**：包含所有转移概率的矩阵

[[https://www.youtube.com/watch?v=rHdX3ANxofs|Youtube解释视频]]


# HMM（Hidden Markov Model）

隐藏状态序列（不可观测）:
    S₁ → S₂ → S₃ → S₄ → ...
    ↓    ↓    ↓    ↓
    O₁   O₂   O₃   O₄   ...
观测序列（可观测）:

例如：
隐藏状态: \[健康, 健康, 感冒, 感冒, 健康, ...\]
观测:    \[正常, 正常, 打喷嚏, 发烧, 正常, ...\]

**HMM的三个基本问题**
- 评估（Evaluation）
	- 给定HMM模型和观测序列，计算该观测序列出现的概率
	- 算法：前向算法（Forward Algorithm）
- 解码（Decoding）
	- 给定HMM模型和观测序列，找出最可能的隐藏状态序列
	- 算法：维特比算法（Viterbi Algorithm）
- 学习（Learning）
	- 给定观测序列，学习HMM的参数
	- 算法：Baum-Welch算法（前向-后向算法）