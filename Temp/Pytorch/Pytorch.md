---
date: 2025-07-28
tags:
  - FIT5215
author:
  - Siyuan Liu
aliases:
  - Summary
---
# 关于torch包下所有的操作,直接查看官方doc
[The Official DOC](https://docs.pytorch.org/docs)

# 一些格外需要注意的细节
1. `torch.tensor()` always copies `data`, which means If you have a Tensor `data` and just want to change its `requires_grad` flag, use `requires_grad_()` or `detach()` to avoid a copy. If you have a `numpy array` and want to avoid a copy, use `torch.as_tensor()`


# `torch.tensor`

## `requires_grad=True`
**作用**: 告诉`PyTorch`的自动求导引擎 ([`torch.autograd`](https://docs.pytorch.org/docs/stable/autograd.html#module-torch.autograd "torch.autograd"))：“请开始追踪对这个张量（Tensor）所做的所有操作. `PyTorch` 会在内存中构建一个**计算图 (Computational Graph)**, 记录下所有与它相关的数学运算，以便后续能够自动计算梯度”

# `torch.nn`
`torch.nn` 的核心思想是将神经网络的各个部分封装成独立的**模块 (Module)**. 一个层是模块，一个损失函数是模块，甚至整个网络本身也是一个大模块，这个大模块由许多小模块组成.
[`torch.nn`](https://docs.pytorch.org/docs/stable/nn.html)

# `torch`的方法
### `torch.mean()` 
requires tensors to be in `torch.float32` or it throws errors
### `torch.reshape(input, shape)`
shape参数可以是多维数据, 假如某一维度值为-1, 说明需要根据确定的其他维度自动重构shape
==Attention:==
1. **返回视图 (View) - 不创建新对象**： 如果原始张量是**内存连续的 (contiguous)**，并且改变形状后的张量也能维持这种连续性，那么 `reshape()` **不会**创建新的数据副本。它会返回一个指向原始数据的新“视图 (view)”。这个新的张量对象会与原始张量共享底层的数据存储。这意味着，如果你修改了其中一个张量的数据，另一个也会随之改变。你可以用 `torch.is_storage_shared(x, y)` 来检查两个张量是否共享底层存储。
    
2. **创建副本 (Copy) - 创建新对象**： 如果原始张量**不是内存连续的**（例如，你对它做了 `transpose()` 或切片等操作），那么 `reshape()` **必须**在内存中复制数据来创建一个新的、连续的张量。在这种情况下，它会创建一个全新的张量对象，与原始张量不共享数据。


### `torch.stack(tensors, dim=0)`
### `toch.permute(input, dims)`
==Attention:==
`torch.permute` **不会**复制数据，而是返回一个共享原始数据存储的**视图**

# Computational Graph
# 数据格式标准
```
Filters: [output_channel = num_of_filters, input_channel = filter_depth, filter_height, filter_width]
```
    
```
Data batch: [batch_size, input_channel, input_height, input_width]
```


# Function
## detach

|**特性**|**x.detach()**|**with torch.no_grad():**|
|---|---|---|
|**作用范围**|**局部**：只针对调用它的那个 Tensor。|**全局（上下文）**：包裹在该代码块内的**所有**计算。|
|**显存占用**|**较高**：虽然切断了联系，但之前的计算图中间状态可能还保留在内存中。|**最低**：根本不建立计算图，不保存中间激活值，大幅节省显存。|
|**主要目的**|**截断梯度流**：用于复杂的网络设计（如 GAN、RL），只想更新部分网络。|**推理/评估 (Inference)**：验证模型或测试模型时，完全不需要反向传播。|
|**结果属性**|返回一个新的 Tensor，`requires_grad=False`。|代码块内生成的所有 Tensor，默认 `requires_grad=False`。|
## expand

## repeat
## squeeze
