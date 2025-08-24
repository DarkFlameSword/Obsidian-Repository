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

# Regularization Techniques
## Regularization related to Weight
### L1 / L2 Regularization
## Regularization related to Construction
### Dropout
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
## Regularization related to Data
### Data Augmentation
**理解:**
在不额外收集新数据的情况下，通过对已有数据进行变换，来人工增加训练数据的多样性的一种方法

**例子:**
- 图像数据

	- **几何变换**
    
	    - 翻转（horizontal / vertical flip）
	        
	    - 旋转（rotation）
	        
	    - 平移（translation）
	        
	    - 缩放（scaling, zoom）
	        
	    - 剪裁（random crop, center crop）
        
	- **颜色与光照调整**
	    
	    - 随机亮度、对比度、饱和度变化
	        
	    - 色彩抖动（color jitter）
	        
	    - 灰度化
	        
	- **噪声与模糊**
	    
	    - 加入高斯噪声
	        
	    - 模糊处理（Gaussian blur, motion blur）
	        
	- **高级方法**
	    
	    - **Cutout**：随机遮挡部分区域
	        
	    - **Mixup**：将两张图像混合
	        
	    - **CutMix**：将一张图像的部分区域替换为另一张
        

- 文本数据

	- 同义词替换（synonym replacement）
	    
	- 随机插入 / 删除 / 交换词语
	    
	- 回译（back translation，例如中译英再译回中）
	    
	- 使用预训练模型生成增强句子（如 GPT 生成 paraphrase）
    
- 音频数据

	- 时间拉伸（time stretching）
	    
	- 音调变化（pitch shifting）
	    
	- 加噪声（background noise）
	    
	- 裁剪、拼接
    
- 表格数据

	- 对数值特征加噪声
	    
	- SMOTE（合成少数类过采样技术，用于类别不平衡问题）
**代码:**
```
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(15),      # 随机旋转 ±15°
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor()
])

```

### Label smoothing
### Data mix-up
### Data Cut-mix
## Indirect Regularization
### Batch Normalization
**理解:**
网络每一层的输入分布在训练中不断改变 → 导致训练难以收敛；所以通过BN 等方法规范化输入来缓解`Internal Covariate Shift`

**公式:**
在每一层对输入做标准化：
$$\hat{x} = \frac{x - \mu}{\sigma}$$
- $\mu$: mini-batch 的均值
- $\sigma$: mini-batch 的方差
- $x$: mini-batch 的输入h
- $\hat x$: mini-batch 第一次处理后的输入
$$y = \gamma \hat{x} + \beta$$
- $\gamma$: 可训练参数,保证网络有足够的表达能力
- $\beta$: 可训练参数,保证网络有足够的表达能力
- $y$: mini-batch在BN后的输入

**作用:**
- Cope with internal covariate shift
- Reduce gradient
- vanishing/exploding
- Reduce overfitting
- Make training more stable
- Converge faster

---


## Early Stopping
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

# Transfer Learning
**理解:**
Remove FC layers from the pretrained model, then replace them with a brand-new FC head

**适用场景:**
- **标注数据不足**
    - 训练大模型需要大量数据，但很多任务数据稀缺（例如医学图像、低资源语言）
- **训练成本太高**
    - 从零开始训练深度网络需要大量算力
- **相似任务间知识可迁移**
    - 比如图像特征、语言模型中的词向量，这些都是通用的，可以复用


## Feature Transfer
- 使用在大数据集（如 ImageNet）上预训练好的模型提取特征。
- 在新任务中只训练一个分类器（如 SVM、全连接层）。
## Fine-Tuning
- 加载预训练模型的参数。
- 在新任务上继续训练：
    - **冻结前几层**（保持通用特征，如边缘、颜色），只训练后几层。
    - 或者 **全模型微调**，学习率设置较小。

# Ill-conditioning problem

# Long-term dependencies

# Poor correspondence between local and global structures

# Theoretical limits of optimization (but they usually have little use in practice of deep learning)


