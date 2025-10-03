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
### Traditional way
#### Bag-of-word vector representation
![[Pasted image 20251003105704.png]]
**总结：**
- **原理**：将文本表示为词汇的集合，忽略语法和词序
- **优点**：简单易实现
- **缺点**：丢失语义信息，维度高

---
#### TF-IDF (Term Frequency-Inverse Document Frequency)
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
### Word embedding
#### Word2Vec
**理解：**
CBOW（连续词袋）加上 Skip-gram，通过神经网络学习词的分布式表示

**Pretext Task**
![[Pasted image 20251003115511.png]]
**通常会使用的技术：**
- speed up training by sub-sampling (e.g., frequent words)[[Negative sampling]]
- 当词汇表很大（如 100,000 词）时，计算 softmax 需要归一化所有词的概率，计算成本极高[[Hierarchical softmax]]

# 总结
![[Pasted image 20251003120734.png]]