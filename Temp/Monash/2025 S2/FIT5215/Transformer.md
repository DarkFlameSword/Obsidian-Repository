---
date: 2025-10-04
author:
  - Siyuan Liu
tags:
  - FIT5215
---
# Tansformer Types
**作者简笔：**
在你阅读该篇文章时，你必须已经掌握[[Attention]]的相关知识

[Transformer可视化图解](https://poloclub.github.io/transformer-explainer/)
## Classic Transformer
**注意：**
- Encoder，Decoder自身内部的所有的权重和bias在不同的LSTM Cell，QKV Cell， positional encoding，etc.中都是相同的
- Encoder与Decoder内部的所有的权重不同
- ==这样做的目的是为了并行计算，适应不同长度的input和output==

---
### Encoding Input
![[Pasted image 20251004174054.png]]

---
### Decoding Module
![[Pasted image 20251004175552.png]]

---
### Compute the output
![[Pasted image 20251004180248.png]]![[Pasted image 20251004180306.png]]
![[Pasted image 20251004180323.png]]
**完整Transformer结构图：**
![[Pasted image 20251004180447.png]]

**完整Transformer结构图：**
![[Pasted image 20251019200308.png]]

**完整Transformer结构图：**
![[Pasted image 20251019193308.png]]
[Transformer可视化图解](https://poloclub.github.io/transformer-explainer/)

**Batch Norm vs. Layer Norm**：
- **批归一化 (Batch Norm)**：对**一个批次 (batch) 中所有样本**的**同一个特征**进行归一化。它计算的是跨越**批次维度**的均值和方差。它在计算机视觉 (CNNs) 中非常成功，但在处理序列数据（如文本）时效果不佳，因为句子的长度可变，导致批次统计不稳定
- **层归一化 (Layer Norm)**：对**单个样本**的**所有特征**进行归一化。它计算的是跨越**特征维度**的均值和方差，与批次中的其他样本无关。这种特性使它非常适合处理长度可变的序列数据



**总结：**
- Transformer use World Embbedinng to convert words into numbers
- Positional Encoding to keep track of word order
- Self-Attention to keep track of word relationships within the input and output phrases
- Encoder-Decoder Attention to keep tracck of things between the input and output phrases to make sure important words in the inpput are not lost in the translation
- Residual Connection to allow each sub-unit, like Self_Attention, to focus on solving just one part of the problem


## Multiple Heads Transformer
[[Attention#Multiple Heads Transformer]]


---
#### Context Aware Embbeding

---
## Vision Transformer （ViTs）
[[https://arxiv.org/pdf/2010.11929/1000|An image is worth 16x16 words: Transformers for image recognition at scale]]


## Swin Transformer
![[Pasted image 20251005205659.png]]
- ViTs：一张$256 \times 256$的狗狗图片，被划分为$4 \times 4$个patches，每个patch有$16 \times 16$的pixels
- Swin Transformer：与ViTs相比，ST只有最后merging层的tokens是和ViTs一样的，之前几轮数据训练，都是通过将每一个patch不断缩小2倍得到，并且缩放后的target之间互相不连通

![[Pasted image 20251005210642.png]]
- C: the capacity of model(全连接层的参数数量)
# CNNs与Transformer对比
CNNs：
- Locality Sensitive
- Translation Invariant
- Lack of global understanding

Transformer：
- Need very large dataset for training
- Learns inductive biases
- Able to find long-term dependencies