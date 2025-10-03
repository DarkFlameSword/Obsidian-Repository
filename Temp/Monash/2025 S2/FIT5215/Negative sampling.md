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

$$P(w_i) = \frac{f(w_i)^{\frac{3}{4}}} { Σ f(w_j)^{\frac{3}{4}}}
$$
- $f(w_i)$: 词 w_i 的频率
- $\frac{3}{4} 幂次$：平滑处理，降低高频词被过度采样的概率

**代码举例：**
```
# Negative Sampling 示例

# Skip-gram 任务：中心词 "自然" 预测上下文词 "语言"

# 1. 正样本
positive_sample = ("自然", "语言")  # 真实的上下文对
loss_positive = -log(sigmoid(vec("自然") @ vec("语言")))

# 2. 负样本（随机采样）
negative_samples = sample_words(k=5)  # 如: ["苹果", "电脑", "跑步", "音乐", "天空"]
loss_negative = 0
for neg_word in negative_samples:
    loss_negative += -log(sigmoid(-vec("自然") @ vec(neg_word)))

# 3. 总损失
loss = loss_positive + loss_negative
```