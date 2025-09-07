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
![[Pasted image 20250822161854.png]]
## Structure
### Input Layer
**功能:** 
接收原始输入数据

**作用:** 
代表输入数据的特征。 输入层中的每个神经元对应于输入数据的一个特征。 例如，如果输入是图像，则输入层中的每个神经元可能对应于图像的一个像素值

**参数:** 
通常没有参数需要学习。
### Hidden Layers
**功能:** 
对输入数据进行转换和抽象，提取更高级别的特征

**作用:** 是 DNN 的核心部分，负责学习数据中的复杂模式。 每个隐藏层都接收前一层的输出作为输入，并通过激活函数进行非线性转换

**类型:** 
隐藏层可以有很多种类型，例如：
- **Fully Connected Layer/Dense Layer:** 每个神经元都与前一层的所有神经元相连
- **Convolutional Layer:** 使用卷积核对输入数据进行卷积操作，提取局部特征 (常用于图像处理)
- **Recurrent Layer:** 具有循环连接，能够处理序列数据 (常用于自然语言处理)
- **Pooling Layer:** 对输入数据进行降采样，减少特征维度 (常用于图像处理)
- **Activate Function:** 每个隐藏层通常都包含一个激活函数，用于引入非线性，例如 `ReLU`, `sigmoid`, `tanh` 等

**参数:** 
`weights`和`biases`，这些参数在训练过程中通过反向传播算法进行学习

==注意从Hidden Layer 到Output Layer之间的h计算不需要使用激活函数==
### Output Layer
**功能:** 
产生最终的预测结果

**作用:** 
将隐藏层的输出转换为模型所需的格式。 输出层神经元的数量取决于任务的类型
- **分类任务:** 输出层神经元的数量等于类别的数量。 输出层通常使用 softmax 激活函数，将输出转换为概率分布
- **回归任务:** 输出层通常只有一个神经元，输出一个连续值

**参数:** 
`weights`和`biases`，这些参数在训练过程中通过反向传播算法进行学习

**激活函数:**
输出层的激活函数取决于任务的类型
- **分类任务:** `Softmax` (多分类)，`Sigmoid` (二分类)
- **回归任务:** 通常没有激活函数，或者使用线性激活函数
### Optional Layers
 **嵌入层 (Embedding Layer):** 
 将离散的输入 (例如单词) 转换为连续的向量表示 (常用于自然语言处理)
 
**归一化层 (Normalization Layer):** 
对输入数据进行归一化，加速训练并提高模型的泛化能力，例如 Batch Normalization, Layer Normalization

**Dropout 层:** 
在训练过程中随机丢弃一部分神经元，防止过拟合

## Mathematics in Hidden Layer
**神经网络层的定义:**
$$\bar{h} = xW + b \quad \text{and} \quad h = \sigma(\bar{h})$$
- $h = \sigma(xW + b)$
- 其中 `σ` 是激活函数 (activation function)

**使用链式法则求导:**
$$\frac{\partial h}{\partial x} = \frac{\partial h}{\partial \bar{h}} \frac{\partial \bar{h}}{\partial x} = \text{diag}(\sigma'(\bar{h})) W^T \in \mathbb{R}^{4 \times 3}$$
$$\frac{\partial h}{\partial \bar{h}} = 
\begin{bmatrix}
\frac{\partial h_1}{\partial \bar{h}_1} & \frac{\partial h_1}{\partial \bar{h}_2} & \frac{\partial h_1}{\partial \bar{h}_3} & \frac{\partial h_1}{\partial \bar{h}_4} \\
\frac{\partial h_2}{\partial \bar{h}_1} & \frac{\partial h_2}{\partial \bar{h}_2} & \frac{\partial h_2}{\partial \bar{h}_3} & \frac{\partial h_2}{\partial \bar{h}_4} \\
\frac{\partial h_3}{\partial \bar{h}_1} & \frac{\partial h_3}{\partial \bar{h}_2} & \frac{\partial h_3}{\partial \bar{h}_3} & \frac{\partial h_3}{\partial \bar{h}_4} \\
\frac{\partial h_4}{\partial \bar{h}_1} & \frac{\partial h_4}{\partial \bar{h}_2} & \frac{\partial h_4}{\partial \bar{h}_3} & \frac{\partial h_4}{\partial \bar{h}_4}
\end{bmatrix}
=
\begin{bmatrix}
\sigma'(\bar{h}_1) & 0 & 0 & 0 \\
0 & \sigma'(\bar{h}_2) & 0 & 0 \\
0 & 0 & \sigma'(\bar{h}_3) & 0 \\
0 & 0 & 0 & \sigma'(\bar{h}_4)
\end{bmatrix}
= \text{diag}(\sigma'(\bar{h}))$$
$$\frac{\partial \bar{h}}{\partial x} = W^T$$
# Mini-batches

**使用步骤:**
1. 将一个epoch分为很多个Mini-batches
2. 将每一个Mini-batch当作input传入模型
3. 对一个Mini-batch中