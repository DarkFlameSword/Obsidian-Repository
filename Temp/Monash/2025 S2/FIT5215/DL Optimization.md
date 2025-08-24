---
date: 2025-08-24
author:
  - Siyuan Liu
tags:
  - FIT5215
aliases:
  - summary
---
![[Pasted image 20250824192446.png]]

# weight initialization
## What is a good weight/filter initialization?
- Break the ‘symmetry’ of the network: two hidden nodes with the same input should have different weights
    - Large initial weights has better symmetry breaking effect, help avoiding losing signals and redundant units, but could result in exploding values during back-ward and forward passes, especially in Recurrent Neural Networks
- the gradient will not vanishing or exploding
## Xavier Weight Initialization
**作用:**
Try to ensure the variance of the outputs of each layer equal to the
variance of its input. This way, signals and gradients don't shrink or amplify layer by layer in the network

**计算步骤:**
假设某一层有：
- 输入单元数：n_{in}
- 输出单元数：n_{out}
权重矩阵 W 的元素希望满足：
$$Var(W x) \approx Var(x)$$

Xavier 初始化给出了一个简单公式：
- 对**均匀分布**：
$$W \sim U\left[
-\sqrt{\frac{6}{n_{in} + n_{out}}},\sqrt{\frac{6}{n_{in} + n_{out}}}
\right]$$

- 对**正态分布**：
$$W \sim N\left(0, \frac{2}{n_{in} + n_{out}}\right)$$

- ​ $n_{in}$是输入节点数
- $n_{out}$是输出节点数
**适应场景:**
- `sigmoid`, `tanh`
    - 因为这两个函数在输入较大时会饱和，容易导致梯度消失
- 不适用`ReLU`

![[Pasted image 20250824184323.png]]

## He Weight Initialization
**作用:**
Ensure the variance of the outputs of each layer equal to the variance of its inputs, but `He` specially optimized `ReLU`

**Why?:**
Xavier 初始化假设激活函数近似**线性**，但 ReLU 并非对称线性函数，特别是它会把负数全部置零，这会改变输出的方差。因此，需要针对 ReLU 设计新的初始化方式

**计算步骤:**
假设某一层有：
- 输入单元数：n_{in}
- 输出单元数：n_{out}
`ReLU` 的特点是：
$$\text{ReLU}(x) = \max(0,x)$$

大约 **一半的输入会被置为 0**，因此输出的方差会减半。为了保证输出的方差和输入相同，我们需要在初始化时把方差放大一点：
- 对**均匀分布**：
$$W \sim U\left[
-\sqrt{\frac{6}{n_{in}}},\sqrt{\frac{6}{n_{in}}}
\right]$$

- 对**正态分布**：
$$W \sim N\left(0, \frac{2}{n_{in}}\right)$$

- ​ $n_{in}$是输入节点数
- $n_{out}$是输出节点数
==分母是 2 倍，因为 `ReLU` 会丢掉一半的信号==

**适应场景:**
- `ReLU`, `ReLU的所有变种`
- 深层卷积神经网络 / 前馈网络 都可以用 He 初始化

![[Pasted image 20250824185407.png]]

---

# Early Stopping
![[Pasted image 20250824211637.png]]
**理解:**
在验证集表现最好的时候停下来

**代码:**
```
import torch

class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = model.state_dict()  # 保存最佳权重
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# ===== 使用示例 =====
early_stopping = EarlyStopping(patience=5)

for epoch in range(100):
    train(...)   # 训练
    val_loss = validate(...)  # 验证
    
    early_stopping(val_loss, model)

    if early_stopping.early_stop:
        print("Early stopping!")
        model.load_state_dict(early_stopping.best_model_state)  # 恢复最佳参数
        break
```
# Dropout
![[Pasted image 20250824212935.png]]
**理解:**
In each iteration, at each layer, randomly choose some neurons and drop all connections from these neurons

**代码:**
```
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    # 常用 0.2–0.5
    # 输入层一般用小一些,如 0.1–0.2
    # 隐藏层常用 0.5 左右
    nn.Dropout(p=0.5),  
    
- 隐藏层常用 0.5 左右
    nn.Linear(64, 10)
)

```
---
# Batch Normalization
**理解:**

# Ill-conditioning problem

# Long-term dependencies

# Poor correspondence between local and global structures

# Theoretical limits of optimization (but they usually have little use in practice of deep learning)


