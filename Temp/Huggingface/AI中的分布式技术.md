---
date: 2025-10-12
author:
  - Siyuan Liu
tags:
  - Huggingface
---
# Tensor Parallelism
通过将模型内部的**单个巨大张量（权重矩阵）** 切分到多个 GPU 上，来解决这个问题。这样，每个 GPU 只需存储和计算矩阵的一部分，从而协同完成整个运算。

# Data Parallelism
每个 GPU 上都有模型的完整副本，但处理不同批次的数据


# Pipeline Parallelism
将模型的不同层（Layers）分配到不同的 GPU 上
它将整个模型的**连续多层 (a block of layers)** 作为一个整体，分配到不同的计算设备（GPU）上。每个 GPU 负责模型的一部分计算。

- **GPU 0** 可能负责：词嵌入层 + Transformer 第 0-7 层
    
- **GPU 1** 可能负责：Transformer 第 8-15 层
    
- **GPU 2** 可能负责：Transformer 第 16-23 层
    
- **GPU 3** 可能负责：Transformer 第 24-31 层 + 最终的归一化层和输出层