---
date: 2025-11-09
author:
  - Siyuan Liu
tags:
  - FIT5215
---
BERT 的核心架构并不复杂，它基本上就是一个**多层堆叠的 Transformer Encoder**（编码器）
- 与 GPT（使用 Transformer Decoder）不同，BERT 放弃了生成能力，专注于“理解”文本
- 它利用 Transformer 的**自注意力机制（Self-Attention）**，能够同时处理输入序列中的所有词，从而捕捉词与词之间复杂的长距离依赖关系

# Input Process
BERT 的输入不仅仅是词向量，它采用了独特的“三合一”嵌入层相加的方式：
1. **Token Embeddings（词嵌入）**：单词本身的向量表示
    - 引入了两个特殊标记：`[CLS]`（放在句首，用于表示整个句子的语义）和 `[SEP]`（用于分隔两个句子）
2. **Segment Embeddings（段嵌入）**：用来标记当前词属于句子 A 还是句子 B（用于处理像问答这样的双句任务）
3. **Position Embeddings（位置嵌入）**：因为 Transformer 并行处理所有词，没有时序概念，所以必须人为加入位置信息来告诉模型单词的顺序

最终输入 = 词嵌入 + 段嵌入 + 位置嵌入
# Pretrain
## Task 1: Masked words prediction
- 15% of the words are masked at random
- Not all tokens were masked in the same way. Given a masked word, it happens
	- With an 80% of chance, this word is replaced by the **MASK** token
	- With a 10% of chance, this word is replaced by a random word
	- With a 10% of change, this word is left intact
- Predict the indices of masked words on top of representations of those words.
## Task 2: Next sentence prediction
- When choosing the sentences A and B for each pretraining example, 50% of the time B is the actual next sentence following A (labeled as IsNext), and 50% of the time it is a random sentence from the corpus (labeled as NotNext).
- The ℝ final hidden vector of the special **CLS** token as $C \in \mathbf{R}^H$ is used to predict two labels: IsNext and NotNext.

# Fine-tuning
在具体的下游任务（如情感分析、命名实体识别）上，使用少量的标注数据，对预训练好的 BERT 参数进行轻微调整。通常只需要在 BERT 输出层加一个简单的线性分类器即可。