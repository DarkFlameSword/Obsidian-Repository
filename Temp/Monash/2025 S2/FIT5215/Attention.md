---
date: 2025-10-05
author:
  - Siyuan Liu
tags:
  - FIT5215
---
# 作者简笔
关于注意力机制，从Monash FIT5215 提供的PPT教案来看是脱离实际的，建议结合网络教学资源。
例如：[[https://www.youtube.com/watch?v=PSs6nxngL6k&list=PLJI03vBSNJqQQVt-6T4dHcy0V_R0p3gL7&index=26|# Attention for Neural Networks, Clearly Explained!!!]]

![[Pasted image 20251004165159.png]]

# Attention Mechanism Type
## Global attention
Use all input hidden states of the encoder when deriving the context $c_t$
在解码器（Decoder）生成每一个词时，都要计算当前解码状态与编码器（Encoder）**所有**隐藏状态的对齐分数（attention score）

---
## Local attention
Use a selective window of input hidden states of the encoder when deriving the context  $c_t$
为了降低计算成本，我们假设在生成某个词时，只需要关注输入序列中的一个**小的“窗口”区域**，而不是全部

---
## Self-Attention
#### Query, Key, Value
![[Pasted image 20251004172403.png]]
**理解：**
1. 并行先计算每一个token的Q，K，V
2. 计算目标token的Q与其他所有token的Cosine Similarity（normally use dot product）
3. 使用Softmax，将目标token的Q与其他所有token的Cosine Similarity，转化为概率分布
4. 使用步骤3得出的对应概率分布，缩放每一个Value
5. 将缩放后的值相加得到encoding的context vector

==注意==
- 首先需要明白[[Positional Encoding]]的概念

## Multiple Heads Transformer
![[Pasted image 20251004173614.png]]
**计算过程：**

**分头 (Split)**： 
模型不会用一个大的512维向量来表示一个词。相反，它会把这个大向量**切分**成多个小的部分（比如8个64维的小向量）。然后，为每个“头”都创建一套独立的查询（Query）、键（Key）、值（Value）矩阵。这就好比给每个专家发一套专属的分析工具
    
**并行计算注意力 (Parallel Attention Calculation)**： 
这8个“头”**并行工作**，互不干扰。每个头都拿着自己的Q, K, V工具，独立地计算一遍注意力分数。因为每个头的工具（权重矩阵）是随机初始化的，并且在训练中独立更新，所以它们会逐渐学会关注输入序列中**不同方面**的信息
- 一个头可能学会了关注语法依赖
- 另一个头可能学会了关注动词和宾语的关系
- 还有一个头可能学会了关注相近词的模式

**合头 (Concatenate & Combine)**：
将8个头各自计算出的结果（8个64维的向量）**拼接**起来，形成一个大的向量（8 * 64 = 512维）。最后，通过一个额外的全连接层（Linear Layer）对这个拼接后的大向量进行整合和降维，得到最终的输出。这就好比一个项目经理把所有专家的报告汇总起来，写出一份最终的、统一的结论


---
## Masked Self-Attention
![[Pasted image 20251005202445.png]]
在计算Q和K的Cosine Similarity的时候，需要将当前目标token与未来token的Cosine Similarity设置为0

## Cross-Attention
**理解：**
与Self-Attention不同的是，Cross-Attention的Q来源于target sequence，K和V来源于conditional Information（通常是用户输入的text information）

## Pyramidal encoders

## Hierarchical Attention

## Soft/Hard Attention



