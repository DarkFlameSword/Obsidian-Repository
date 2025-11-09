---
date: 2025-11-09
author:
  - Siyuan Liu
tags:
  - FIT5215
---
# ViTs: Fine Tunning
### 1. 准备微调结构

微调时，我们需要对预训练模型的结构做一点小小的改造：

- **替换 MLP Head**：
    - 预训练时的 MLP Head（分类头）是针对原任务的（例如 ImageNet-21k 的 21,843 个类别）
    - 微调时，我们把它去掉，换成一个新的、零初始化的**线性层**（Linear Layer），其输出节点数等于你下游任务的类别数（例如，如果是猫狗分类，就设为 2）
- **保留 Encoder 主体**：Transformer Encoder 部分的所有参数（权重）都保留并作为微调的初始状态

### 2. 处理更高分辨率的输入 (关键技术点)

ViT 论文发现，在微调时使用比预训练时**更高分辨率**的图片，通常能显著提升效果。

- **问题**：ViT 处理图片的方式是把它切成固定大小的 Patch（例如 $16 \times 16$）。如果图片变大了，Patch 的数量（序列长度）就会增加
- **挑战**：Transformer 可以处理变长的序列，但是预训练好的**位置编码（Position Embeddings）**是固定长度的，这就对不上了
- **解决方案：2D 插值（2D Interpolation）**：
    - ViT 根据预训练好的位置编码在原始图片中的大概位置，对它们进行二维插值（通常是双线性插值），从而“拉伸”出适应新序列长度的新位置编码
    - _这是 ViT 微调中唯一需要手动调整网络结构参数的地方

### 3. 微调训练策略

微调不仅仅是接着训练，通常需要采用一些特定的策略：

- **较小的学习率**：使用比预训练时小得多的学习率（例如小 10 倍或更多），以免破坏预训练学到的良好特征表示
    - 通常使用带动量的 SGD 优化器（不同于预训练时常用的 AdamW）
- **全参数微调 vs. 部分微调**：
    - 虽然可以只训练新的 MLP Head（冻结 Encoder），但通常**全参数微调**（更新网络所有层的权重）效果最好
- **训练步数**：微调所需的 epoch 数通常远少于预训练

---
# Fine-Tuning with Additional Components
insert to a pre-trained ViT some additional components that favour the computations of ViT and fine-tune them
## Fine-Tuning with Prompts
**什么是 Visual Prompt？**

在 NLP 中，Prompt 是一些加在输入文本前的额外单词（例如，把输入“这部电影很好看”变成“翻译成英文：这部电影很好看”），用来“提示”模型要做什么任务。

在 ViT 中，**Visual Prompts（视觉提示）** 是一组**可学习的向量（Learnable Vectors）**
- 它们和图像本身的 Patch Embeddings 长得一样（维度相同）
- 它们不包含任何图像信息，其初始值可能是随机噪声
- 在输入 Transformer Encoder 之前，这些 Prompt 向量被拼接到原始的图像 Patch 序列中

### 工作流程 (VPT - Visual Prompt Tuning)

以最具代表性的 **VPT (Visual Prompt Tuning)** 为例，其工作流程如下：

1. **准备输入**：
    - 将图像切分为 Patch 并转化为 Patch Embeddings序列：$E = [e_1, e_2, \dots, e_N]$
    - 准备预训练好的 Class Token：`[CLS]`
    - **引入 Prompts**：初始化 $P$ 个可学习的向量 $P = [p_1, p_2, \dots, p_P]$
2. **拼接序列**：
    - 将它们拼成一个新的、更长的输入序列：
    - `Input = [[CLS], p_1, p_2, ..., p_P, e_1, e_2, ..., e_N]`
    - _注意：Prompt 向量通常放在 [CLS] token 和真实的图像 Patch 之间（具体位置可变）
3. **前向传播与训练**：
    - 整个序列通过冻结的 ViT Encoder
    - 在训练时，**只有 Prompt 向量 $P$ 和最后的线性分类头（Head）的参数会被更新**。ViT 原本那几亿个参数完全不动

---
## Fine-Tuning with Adapters
**Model Fine-Tuning with Adapters**（适配器微调）是另一种非常流行的参数高效微调（PEFT）技术。与 Prompt Tuning 在输入端“做文章”不同，Adapter Tuning 的思路是在预训练模型的内部结构中“插入”一些小型的、可学习的神经网络模块（称为 **Adapters**）
在微调时，我们冻结原有的巨量模型参数，只训练这些新插入的 Adapter 模块

**什么是 Adapter？**

Adapter 本质上是一个**Bottleneck Architecture**的小型神经网络。它通常由两个线性层（全连接层）和一个非线性激活函数组成：
1. **降维层（Down-projection）**：将输入的高维特征向量（维度为 $d$）压缩到一个低维空间（维度为 $m$，其中 $m \ll d$）
2. **非线性激活（Non-linearity）**：如 ReLU 或 GELU，增加非线性能力
3. **升维层（Up-projection）**：将低维特征重新还原回原始的高维空间（维度 $d$）

此外，Adapter 通常还会包含一个**残差连接（Residual Connection）**，即 Adapter 的输出会与它的输入相加。这保证了在 Adapter 初始状态（或参数接近 0 时），模型可以退化为原始模型，不会破坏已有的知识。

$$Output = Adapter(Input) + Input$$

**Adapter 放在哪里？**

在 Transformer 架构（如 BERT 或 ViT）中，Adapter 模块通常被插入到每一层 Transformer Block 的关键位置。常见的插入点有两个：
1. **在多头自注意力（Multi-Head Attention）层之后**
2. **在大多数前馈神经网络（Feed-Forward Network, FFN）层之后**

这意味着如果一个模型有 12 层，每层有 2 个插入点，那么总共就会插入 24 个 Adapter 模块。

### 3. 工作流程

1. **初始化**：加载预训练好的大模型（如 ViT-L/16），并将其所有原始参数**冻结**
2. **插入**：在预定义的位置插入随机初始化的 Adapter 模块
3. **训练**：在下游任务数据上进行训练。梯度只会更新 Adapter 模块的参数（以及最终的分类头），原模型的参数保持不变

---
## Fine-Tuning with LoRA
**LoRA (Low-Rank Adaptation)** 是一种目前非常火爆且高效的参数高效微调（PEFT）技术。它的核心思想与 Adapter 有些相似（都是冻结原模型，引入少量新参数），但它实现的方式更加巧妙和数学化：它不插入新的模块层，而是通过**低秩分解**来模拟参数的更新量。

### 1. 核心直觉：大模型的“低内在维度”

LoRA 的提出基于一个重要的学术发现：预训练大模型虽然参数极多（例如 GPT-3 有 1750 亿参数），但在处理特定下游任务时，其权重矩阵的更新量实际上并不需要那么高的“自由度”。也就是说，**权重更新矩阵是“低秩”的**（Low-Rank）

换句话说，为了适应新任务，我们不需要在一个巨大的高维空间里四处乱撞，只需要在一个非常小的子空间里找方向就够了

### 2. LoRA 的工作原理

假设在大模型中某一层（比如 Transformer 的 Attention 层的 Query 投影矩阵）的原始权重矩阵是 $W_0$，它的维度是 $d \times k$

在全参数微调时，我们会学到一个更新矩阵 $\Delta W$，微调后的权重是 $W = W_0 + \Delta W$。这个 $\Delta W$ 的维度也是 $d \times k$，参数量非常大

LoRA 的做法是：

冻结 $W_0$，把巨大的 $\Delta W$ 分解为两个非常小的矩阵 $A$ 和 $B$ 的乘积：

$$\Delta W = B \cdot A$$

- 矩阵 $A$ 的维度是 $r \times k$
- 矩阵 $B$ 的维度是 $d \times r$
- **$r$ 是秩（Rank）**，是一个非常小的超参数（例如 $r=4, 8, 16$），远小于 $d$ 和 $k$（通常是几千）

**训练过程：**

- 输入 $x$ 过来时，它同时走两条路：
    1. 走原模型的老路：$h_1 = W_0 \cdot x$
    2. 走 LoRA 的旁路：$h_2 = B \cdot (A \cdot x)$
- 最终输出是两者的叠加：$h = h_1 + h_2 = W_0 x + B A x = (W_0 + BA)x$
- 在训练时，我们**只更新 $A$ 和 $B$**，冻结 $W_0$

### 3. LoRA 的独特优势

- **零推理延迟（Zero Inference Latency）**：这是 LoRA 相比 Adapter 最大的优势
    - Adapter 在推理时必须保留那些额外的模块层，这会稍微增加计算量和延迟
    - LoRA 在训练完后，可以直接把学到的 $B \cdot A$ 加回到原权重 $W_0$ 上（即 $W_{new} = W_0 + BA$）。这样，在部署推理时，模型结构和原始模型**完全一样**，没有任何额外的计算开销
- **参数极少**：由于 $r$ 很小，需要训练的参数量（$d \times r + r \times k$）远远小于原来的 $\Delta W$（$d \times k$）
- **训练稳定**：通常矩阵 $A$ 使用高斯初始化，矩阵 $B$ 初始化为 0。这样在训练刚开始时，旁路输出为 0，模型完全等价于预训练模型，训练过程非常平稳。