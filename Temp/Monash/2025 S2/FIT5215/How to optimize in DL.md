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
# Overview of Optimization problem in ML and DL
![[Pasted image 20250908142412.png]]
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
损失函数原本是：

$$J(\theta) = \frac{1}{N}\sum_{i=1}^N l\big(f(x_i;\theta), y_i\big)$$

加正则化后变成：

$$J(\theta) = \frac{1}{N}\sum_{i=1}^N l\big(f(x_i;\theta), y_i\big) + \lambda R(\theta)$$

其中：
- $\lambda > 0$：正则化强度系数
- $R(\theta)$：正则化项
#### L1 Regularization
$$R(θ)=||\theta||=\sum_j∣θ_j​∣$$
- **特点**：鼓励参数变为 0，得到**稀疏模型**（很多权重为 0）
- **用途**：特征选择（自动把不重要的特征权重压到 0）
#### L2 Regularization
$$R(θ)=||\theta||^2​=\sum_j \sqrt{θ_j^2}​$$

- **特点**：惩罚大权重，鼓励参数更均匀分布，不会直接变成 0
- **用途**：常用来防止过拟合
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
$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma+\epsilon}}$$
- $\mu$: mini-batch 的均值
- $\sigma$: mini-batch 的方差
- $\epsilon$: a small value such as $1e^{-7}$
- $x$: mini-batch 的输入h
- $\hat x$: mini-batch 第一次处理后的输入
$$y = \gamma \hat{x} + \beta$$
- $\gamma$: 可训练参数,保证网络有足够的表达能力
- $\beta$: 可训练参数,保证网络有足够的表达能力
- $y$: mini-batch在BN后的输入

**注意:**
- In training phase, it uses the batch statistics**批次统计量** (mean and std)
- In testing phase, it uses the population statistics**总体统计量** (mean and std)

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
**特点:**
- 使用在大数据集（如 ImageNet）上预训练好的模型提取特征
- 在新任务中只训练一个分类器（如 SVM、全连接层）

**代码**:
```
# pip install torch torchvision scikit-learn
import torch, numpy as np
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1) 预处理与数据
tfm = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                          transforms.ToTensor(), transforms.Normalize(
                          mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
train_set = datasets.ImageFolder("data/train", tfm)
val_set   = datasets.ImageFolder("data/val",   tfm)
train_loader = DataLoader(train_set, batch_size=64, shuffle=False, num_workers=2)
val_loader   = DataLoader(val_set,   batch_size=64, shuffle=False, num_workers=2)

# 2) 预训练模型做特征提取（去掉最后分类层）
backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
backbone.fc = nn.Identity()   # 直接输出全局特征
backbone.eval().cuda()

def extract_feats(loader):
    X, y = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            feats = backbone(imgs.cuda()).cpu().numpy()
            X.append(feats); y.append(labels.numpy())
    return np.vstack(X), np.concatenate(y)

Xtr, ytr = extract_feats(train_loader)
Xva, yva = extract_feats(val_loader)

# 3) 只训练一个轻量分类器（逻辑回归 / 也可 SVM）
clf = LogisticRegression(max_iter=2000, n_jobs=-1)
clf.fit(Xtr, ytr)
pred = clf.predict(Xva)
print("Val acc:", accuracy_score(yva, pred))

```
## Fine-Tuning
**步骤:**
![[Pasted image 20250824221050.png]]
1. Freeze all CONV layers in the network
2. Only allow the gradient to backpropagate through the FC layers. Doing this allows our network to warm up(1-5 epoch)
![[Pasted image 20250824221458.png]]
3. unfreeze all layers in the network
4. Continue training the entire network, but with a very small learning rate
5. We do not want to deviate our CONV filters dramatically. Training is then allowed to continue until sufficient accuracy is obtained

**特点:**
- 加载预训练模型的参数。
- 在新任务上继续训练：
    - **冻结前几层**（保持通用特征，如边缘、颜色），只训练后几层
    - 或者 **全模型微调**，学习率设置较小

**代码**:
```
import torch
import torchvision.models as models
import torch.nn as nn

# 加载预训练的 ResNet18
model = models.resnet18(pretrained=True)

# 冻结前面的层
for param in model.parameters():
    param.requires_grad = False

# 替换最后一层分类器（假设新任务有 10 类）
model.fc = nn.Linear(model.fc.in_features, 10)

# 现在只训练最后一层

```

# Ill-conditioning problem

# Long-term dependencies

# Poor correspondence between local and global structures

# Theoretical limits of optimization (but they usually have little use in practice of deep learning)


