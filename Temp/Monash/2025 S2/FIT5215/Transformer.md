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
![[Pasted image 20251109134251.png]]

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
## Vision Transformers （ViTs）
![[Pasted image 20251109134826.png]]
![[Pasted image 20251109134913.png]]
[[Perceptron#Multi-Layer Perceptron|多层感知机MLP]]
[Vision Transformer](https://www.youtube.com/watch?v=vJF3TBI8esQ)
[An image is worth 16x16 words: Transformers for image recognition at scale](https://arxiv.org/pdf/2010.11929/1000)


## Swin Transformer
![[Pasted image 20251005205659.png]]
- ViTs：一张$256 \times 256$的狗狗图片，被划分为$4 \times 4$个patches，每个patch有$16 \times 16$的pixels
- Swin Transformer：与ViTs相比，ST只有最后merging层的tokens是和ViTs一样的，前一轮数据训练，通过将原始图片划分为4个part（$\text{每个part长宽缩小为一半}\frac{1}{2}\times H,\frac{1}{2}\times W$），4个part独立计算patch，$\text{每个patch长宽缩小为一半}\frac{1}{2}\times h,\frac{1}{2}\times w$,并且某一个part数据不互通
![[Pasted image 20251005210642.png]]
- C: the capacity of model(全连接层的参数数量)

### Window Self-Attention in Swin Transformers
为了解决part与part之间信息不互通的问题，引入了**Window Self-Attention**

**抽象理解：**
老师在进行完一轮“小组讨论”后，会让学生们进行一次**轮换**

1. **第一轮 (W-MSA)**：学生们在固定的**小组（Window）**内讨论。
2. **第二轮 (SW-MSA)**：老师说：“请大家向右和向下各移动半个小组的距离，然后重新组成 4 人小组！”
    - 现在，新的小组就形成了。原本在 A 组角落的学生，现在可能会和原来 B 组、C 组、D 组的同学组成一个全新的小组
    - 通过这次**窗口移动 (Shifted Window)**，原本孤立的小组边界被打破了，信息得以在不同的小组之间传递
![[Pasted image 20251019213525.png]]
(上图：窗口发生位移，形成了新的窗口，实现了跨窗口的信息交流)_

Swin Transformer 在连续的网络层中，会**交替**使用这两种注意力机制：
- 第 L 层：使用**常规窗口 (W-MSA)** 进行高效的“小组内”讨论
- 第 L+1 层：使用**移动窗口 (SW-MSA)** 进行“跨小组”讨论


---
# CNNs与ViTs对比
CNNs：
优：
- Translation equivariance
缺：
- Locality Sensitive
- Lack of global understanding
- cannot capture the spatial relationship of local objects inside images

ViTs：
优：
- Able to find long-term dependencies
- Can naturally capture the global information of images
- can find the long-term dependencies among image patches
- ViTs are more robust to patch permutation and occlusion than CNNs
缺：
- Need very large dataset for training
- ViTs 缺乏很多**归纳偏置 (Inductive Bias)**。例如，CNNs 天生就具有“局部性 (locality)”（假设相邻像素更相关）和“平移等变性 (translation equivariance)”（物体在图像中的位置不影响其特征）的先验知识。ViT 没有这些内置假设，它必须从数据中从头学习这些视觉规律。因此，当训练数据不足时，ViT 很难学习到这些基本模式，导致性能不如 CNN
