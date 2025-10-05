---
date: 2025-10-05
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
2. 计算目标token的Q与其他所有token的Cosine Similarity（normally use dot product）
3. 使用Softmax，将目标token的Q与其他所有token的Cosine Similarity，转化为概率分布
4. 使用步骤3得出的对应概率分布，缩放每一个Value
5. 将缩放后的值相加得到encoding的context vector

==注意==
- 首先需要明白[[Positional Encoding]]的概念

---
# # Masked Self-Attention
![[Pasted image 20251005202445.png]]
在计算Q和K的Cosine Similarity的时候，需要将当前目标token与未来token的Cosine Similarity设置为0

# Cross-Attention
**理解：**
与Self-Attention不同的是，Cross-Attention的Q来源于target sequence，Q和V来源于conditional Information（通常是用户输入的text information）