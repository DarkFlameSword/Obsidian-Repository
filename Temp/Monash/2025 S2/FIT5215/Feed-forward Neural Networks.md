---
date: 2025-08-08
author:
  - Siyuan Liu
tags:
  - FIT5215
aliases:
  - summary
---
**定义:** FFNNs 的核心特征是信息单向流动。 每一层接收前一层的输出作为输入，并通过激活函数传递，最终到达输出层。 简单的感知机就是一个 FFNN，它没有隐藏层
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
# DNNs(Deep Neural Networks)
**定义:** DNNs 是一种特殊的 FFNN，其显著特征是拥有大量的隐藏层。 这些额外的层使 DNN 能够学习输入数据中更复杂、抽象的表示。 这种深度是 DNN 能够胜任图像识别、自然语言处理等复杂任务的关键