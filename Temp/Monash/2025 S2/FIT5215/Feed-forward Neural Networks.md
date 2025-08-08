---
date: 2025-08-08
author:
  - Siyuan Liu
tags:
  - FIT5215
aliases:
  - summary
---

```
import torch
import torch.nn as nn

# 定义一个前馈神经网络
class SimpleFFNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 4)   # 输入3个特征，隐藏层4个神经元
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)   # 输出1个结果

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 输入示例
input = torch.tensor([[1.0, 2.0, 3.0]])
model = SimpleFFNN()
output = model(input)
print(output)
```