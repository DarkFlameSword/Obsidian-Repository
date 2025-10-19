---
date: 2025-10-03
author:
  - Siyuan Liu
tags:
  - FIT5215
---


原理：不计算所有词的概率，只区分正样本和少量负样本
正样本：真实的上下文词
负样本：随机采样的非上下文词（通常 5-20 个）



**代码举例：**
```
# Negative Sampling 示例

# Skip-gram 任务：中心词 "自然" 预测上下文词 "语言"

# 1. 正样本
positive_sample = ("自然", "语言")  # 真实的上下文对
p = sigmoid(vec("自然") @ vec("语言"))

# 2. 负样本（随机采样）
negative_samples = sample_words(k=5)  # 如: ["苹果", "电脑", "跑步", "音乐", "天空"]
loss_negative = 0
for neg_word in negative_samples:
    p = sigmoid(-vec("自然") @ vec(neg_word))
    
# 3.训练模型，并且使得模型能够将(target word, context word)预测为1，而对于其他negative pair则预测为0
```