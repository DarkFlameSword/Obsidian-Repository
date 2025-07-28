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
2. 