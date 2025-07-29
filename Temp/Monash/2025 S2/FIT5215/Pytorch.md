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
`torch.mean()` requires tensors to be in `torch.float32` or it throws errors


