---
date: 2025-10-03
author:
  - Siyuan Liu
tags:
  - FIT5215
---
# 文字分析过程
![[Pasted image 20251003105503.png]]
## Text normalization
- Expanding contractions
	- e.g., isn’t → is not, you’re → you are
- Lemmatization
	- e.g. cars → car, running → run, is → be
- Removing special characters and symbols
	- e.g. !,
- Removing stop words
	- e.g., a, and

---
## Feature Extraction
### Word2Vec

**Pretext Task**
![[Pasted image 20251003115511.png]]
**通常会使用的技术：**
- speed up training by sub-sampling (e.g., frequent words)[[Negative sampling]]
- 当词汇表很大（如 100,000 词）时，计算 softmax 需要归一化所有词的概率，计算成本极高[[Hierarchical softmax]]

#### Skip-gram
==target word predicts context words==

#### Bag-of-word vector representation
==context words predict target word==

![[Pasted image 20251003105704.png]]
**总结：**
- **原理**：将文本表示为词汇的集合，忽略语法和词序
- **优点**：简单易实现
- **缺点**：丢失语义信息，维度高

---
### TF-IDF (Term Frequency-Inverse Document Frequency)
![[Pasted image 20251003112837.png]]

TF: term frequency - number of term occurrences in a document
IDF: inverse document-frequency - how much information the term provides in corpus ?

**数学公式：**
$$\text{TF-IDF} = TF(t,d) × IDF(t)$$
$$IDF(t, C) = log\frac{|C|}{|C_t|}$$
- $|C|$: 文档总数
- $|C_t|$: 包含词项 t 的文档数量（Document Frequency）
==为了避免分母为0，一般使用平滑IDF: $IDF(t, C) = log\frac{|C|+1}{|C_t|+1}$==

**总结：**
- **原理**：词频 × 逆文档频率，衡量词的重要性
- **优点**：考虑词的重要性，降低常见词权重
- **缺点**：仍然是稀疏表示


---
# Words Embedding Processes
## 第 1 步：准备语料库 (Corpus)

这是基础。我们需要大量的文本数据，因为模型的学习原则是“**观其伴而知其义**”(You shall know a word by the company it keeps)。
- **来源**：维基百科、新闻文章、书籍、社交媒体帖子等。语料库越大、质量越高，学习到的词嵌入效果就越好。
- **例子**：我们拿到了一堆关于动物的句子，如 "The cat sleeps on the sofa.", "A dog chases the cat." 等。

## 第 2 步：文本预处理 (Preprocessing)

我们需要把原始的、杂乱的文本数据，整理成模型可以理解的格式。
1. **分词 (Tokenization)**：将句子切分成一个个独立的词元 (token)。
    - `"The cat sleeps on the sofa."` -> `["the", "cat", "sleeps", "on", "the", "sofa"]`
2. **构建词典 (Build Vocabulary)**：统计所有出现过的不重复词元，并为每个词元分配一个唯一的整数 ID
    - `{"the": 0, "cat": 1, "sleeps": 2, "on": 3, "sofa": 4, "a": 5, "dog": 6, "chases": 7, ...}`
    - 词典的总大小就是 `vocab_size`
3. **数据数值化**：将所有句子用对应的 ID 替换
    - `["the", "cat", "sleeps"]` -> `[0, 1, 2]`

## 第 3 步：定义一个“前置”任务 (Pretext Task)

这是整个流程的**核心和精髓**。我们并不直接去“学习向量”，而是设计一个“前置”任务。模型在努力完成这个任务的过程中，会**副产物**式地学到我们真正想要的词嵌入。

Word2Vec 提供了两种经典任务：

- **Skip-gram (跳字模型)**：**根据中心词，预测上下文**。这是更常用的一种
    - **任务**：给模型输入 `cat` (ID: 1)，要求它预测出周围可能出现的词，如 `the` (ID: 0), `sleeps` (ID: 2)
    - **训练数据**：我们会从语料库中生成大量的 `(中心词, 上下文词)` 词对，例如 `(1, 0)`, `(1, 2)`
- **CBOW (连续词袋模型)**：**根据上下文，预测中心词**
    - **任务**：给模型输入上下文 `[the, sleeps]` (IDs: [0, 2])，要求它预测出中间的词是 `cat` (ID: 1)

## 第 4 步：构建神经网络模型

我们构建一个非常简单的浅层神经网络来执行上面定义的“前置”任务。以 **Skip-gram** 为例，模型结构通常是：

1. **输入层 (Input Layer)**：输入中心词的 ID，通常表示为一个 one-hot 向量（一个长度为 `vocab_size` 的向量，只有对应词 ID 的位置是1，其余都是0）
2. **隐藏层 / 嵌入层 (Hidden/Embedding Layer)**：这是一个没有激活函数的全连接层。**这一层就是魔法发生的地方！**
    - 该层的权重是一个巨大的矩阵，形状为 `[vocab_size, embedding_dimension]` (例如 `[10000, 300]`)
    - 当 one-hot 输入向量与这个权重矩阵相乘时，其效果就等同于**直接从矩阵中“抽取”出对应词 ID 的那一行向量**
    - **关键：这个权重矩阵，就是我们最终梦寐以求的“词嵌入矩阵”！** 训练开始时，它的值是随机的
3. **输出层 (Output Layer)**：一个全连接层，输出一个长度为 `vocab_size` 的向量，并通过 Softmax 函数转换成概率分布，表示词典中每个词作为上下文出现的概率

## 第 5 步：训练模型

我们把第3步生成的 `(输入, 目标)` 词对喂给模型，然后执行标准的神经网络训练流程：

1. **前向传播**：输入一个中心词，模型预测其上下文词的概率分布
2. **计算损失**：将模型的预测概率与真实的上下文词进行比较，计算损失（例如使用交叉熵损失函数）
3. **反向传播**：根据损失值，使用优化器（如 SGD）通过反向传播算法更新网络中的所有权重
4. **重复**：对语料库中成千上万的词对重复以上步骤

在训练过程中，模型为了降低损失（即更准确地预测上下文），会不断调整权重。其中，**隐藏层的权重矩阵（词嵌入矩阵）会被迫进行调整**，使得那些经常在一起出现的词（如 "cat" 和 "sleeps"）在向量空间中的表示变得越来越相似

## 第 6 步：提取词嵌入矩阵

训练结束后，这个擅长“猜词游戏”的模型本身对我们来说**通常没什么用**。我们**真正想要的**是它在训练过程中学到的“副产品”
- 我们**丢弃**模型的输出层
- 我们**提取**出训练好的**隐藏层权重矩阵**（那个 `[vocab_size, embedding_dimension]` 的大矩阵）

这个矩阵就是最终的**词嵌入**。矩阵的第 `i` 行，就是词典中 ID 为 `i` 的那个词的密集向量表示。现在，你可以用这个矩阵/查找表，将你语料库中的任何一个词转换成一个包含丰富语义信息的向量了。

---
# 总结
![[Pasted image 20251003120734.png]]
