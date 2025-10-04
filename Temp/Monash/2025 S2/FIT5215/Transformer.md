---
date: 2025-10-04
author:
  - Siyuan Liu
tags:
  - FIT5215
---
# Self-Attention
## Query, Key, Value
![[Pasted image 20251004172403.png]]
**理解：**
1. 并行先计算每一个token的Q，K，V
2. 计算目标token的Q与其他所有token的Cosin Similarity（normally use dot product）
3. 使用Softmax，将目标token的Q与其他所有token的Cosin Similarity，转化为概率分布
4. 使用步骤3得出的对应概率分布，缩放每一个Value
5. 将缩放后的值相加得到encoding的向量值

==注意==
- 首先需要明白[[Positional Encoding]]的概念


# Multiple Heads
![[Pasted image 20251004173614.png]]
# Transformer
**注意：**
- Encoder，Decoder自身内部的所有的权重和bias在不同的LSTM Cell，QKV Cell， positional encoding，etc.中都是相同的
- Encoder与Decoder内部的所有的权重不同
- ==这样做的目的是为了并行计算，适应不同长度的input和output==

---
## Encoding Input
![[Pasted image 20251004174054.png]]

---
## Decoding Module
![[Pasted image 20251004175552.png]]

---
## Compute the output
![[Pasted image 20251004180248.png]]![[Pasted image 20251004180306.png]]
![[Pasted image 20251004180323.png]]
![[Pasted image 20251004180447.png]]
**总结：**
- Transformer use World Embbedinng to convert words into numbers
- Positional Encoding to keep track of word order
- Self-Attention to keep track of word relationships within the input and output phrases
- Encoder-Decoder Attention to keep tracck of things between the input and output phrases to make sure important words in the inpput are not lost in the translation
- Residual Connection to allow each sub-unit, like Self_Attention, to focus on solving just one part of the problem

