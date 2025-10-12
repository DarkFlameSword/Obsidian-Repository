---
date: 2025-10-12
author:
  - Siyuan Liu
tags:
  - Huggingface
---
# 作者简述
本篇文章主要用于记录Qwen3-4B-Instruct-2507模型的源码理解，用关键module的相对路径作为标题

---
# .\transformers\models\qwen3\configuration_qwen3.py
# base_model_tp_plan
在训练或运行像 Qwen3 这样的大语言模型（LLM）时，会遇到一个巨大的挑战：模型的参数（权重）量非常大。一个单一的权重矩阵（例如，用于计算 Q, K, V 的矩阵）就可能达到几十 GB，可能无法完全放入单个 GPU 的显存（VRAM）中。
所以将模型内部的**单个巨大张量（权重矩阵）** 切分到多个 GPU 上，来解决这个问题，也就是张量并行 (Tensor Parallelism, 简称 TP)

```
base_model_tp_plan = {
    "layers.*.self_attn.q_proj": "colwise",
    "layers.*.self_attn.k_proj": "colwise",
    "layers.*.self_attn.v_proj": "colwise",
    "layers.*.self_attn.o_proj": "rowwise",
    "layers.*.mlp.gate_proj": "colwise",
    "layers.*.mlp.up_proj": "colwise",
    "layers.*.mlp.down_proj": "rowwise",
}
```
- - `layers`: 指的是模型的主体部分，即 Transformer 层的列表  
- `*`: 通配符，表示**任意一层**（例如第 0 层、第 1 层...直到最后一层）
- `self_attn`: 指的是该层内的自注意力（Self-Attention）模块
- `q_proj`: 指的是自注意力模块中用于生成 Query（查询）的**投影层 (projection layer)** 的权重
- **`"colwise"` (Column-wise / 按列切分)**
    - 将权重矩阵沿着**列（column）** 的方向进行切分。
    - **数学原理**: 对于矩阵乘法 `Y = XA`，如果我们将矩阵 `A` 按列切分为 `[A_1, A_2, ..., A_n]`，那么输出 `Y` 也可以被自然地切分为 `[XA_1, XA_2, ..., XA_n]`。每个 GPU 计算自己分到的那部分 `XA_i`，得到输出的一部分 `Y_i`。这个过程几乎不需要通信。
- **`"rowwise"` (Row-wise / 按行切分)**
    - 将权重矩阵沿着**行（row）** 的方向进行切分。
    - **数学原理**: 这种方式通常用于一个计算模块的**输出投影层**。它的输入 `X` 通常是前一个 `colwise` 层并行计算得到的、已经被切分的结果。当它与一个按行切分的权重 `A` 相乘后，每个 GPU 会得到一个部分和（partial sum）。最后需要一个**全局规约（All-Reduce）**操作，将所有 GPU 上的部分和加起来，得到最终的、完整的输出结果。这个 `All-Reduce` 操作是必需的通信步骤。

所以，`layers.*.self_attn.q_proj`键的含义是：“模型中**每一层**的**自注意力模块**里的**q_proj权重**”

---
# .\transformers\models\qwen2\tokenization_qwen2_fast.py



---
# .\transformers\models\qwen3\modular_qwen3.py


---
# .\transformers\models\qwen3\modeling_qwen3.py

