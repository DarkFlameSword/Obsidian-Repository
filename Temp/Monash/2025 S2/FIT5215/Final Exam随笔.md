---
date: 2025-11-04
author:
  - Siyuan Liu
tags:
  - FIT5215
---
# PyTorch Module

## 1. nn.Conv2D - 二维卷积层

### 功能
用于对输入进行二维卷积操作，常用于图像处理。

### 基本语法
```python
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
```

### 入参说明
| 参数 | 说明 | 类型 | 默认值 |
|-----|------|------|--------|
| `in_channels` | 输入通道数 | int | 必需 |
| `out_channels` | 输出通道数 | int | 必需 |
| `kernel_size` | 卷积核大小 | int或tuple | 必需 |
| `stride` | 步长 | int或tuple | 1 |
| `padding` | 填充大小 | int或tuple | 0 |
| `dilation` | 膨胀系数 | int或tuple | 1 |
| `groups` | 分组卷积的组数 | int | 1 |
| `bias` | 是否使用偏置 | bool | True |

### 使用示例
```python
import torch
import torch.nn as nn

# 创建卷积层：输入3通道，输出64通道，3x3卷积核
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)

# 输入张量：(batch_size, 3, 224, 224)
x = torch.randn(4, 3, 224, 224)
output = conv(x)
print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")  # (4, 64, 224, 224)
```

### 输出shape计算
$$
W_o = {floor}(\lfloor \frac{W_i - K_w + 2P}{S_w} \rfloor) + 1 \;\;\;\;\;\;\; H_o = floor(\lfloor \frac{H_i - K_h + 2P}{S_h} \rfloor) + 1
$$
- `Wi,Hi`: 输入图像尺寸
- `Wo,Ho`: 输出图像尺寸
- `Kw,Kh`: 卷积核宽, 高
- `Sw,Sh`: 步长核宽, 高
-  `P`: 填充
只有 **TensorFlow/Keras** 的 `Conv2D` 支持 `padding='same'`
$$H_{out}=\frac{H_{in}}{stride}$$


---

## 2. nn.Conv2dTranspose - 转置卷积层（反卷积）

### 功能
用于上采样操作，与卷积相反，常用于图像生成。

### 基本语法
```python
nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, bias=True)
```

### 入参说明
| 参数 | 说明 | 类型 | 默认值 |
|-----|------|------|--------|
| `in_channels` | 输入通道数 | int | 必需 |
| `out_channels` | 输出通道数 | int | 必需 |
| `kernel_size` | 卷积核大小 | int或tuple | 必需 |
| `stride` | 步长 | int或tuple | 1 |
| `padding` | 填充大小 | int或tuple | 0 |
| `output_padding` | 输出填充大小 | int或tuple | 0 |
| `bias` | 是否使用偏置 | bool | True |

### 使用示例
```python
import torch
import torch.nn as nn

# 创建转置卷积层：输入64通道，输出3通道
deconv = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1)

# 输入张量：(batch_size, 64, 56, 56)
x = torch.randn(4, 64, 56, 56)
output = deconv(x)
print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")  # (4, 3, 112, 112)
```

### 输出shape计算
$$\begin{aligned}
&H_{out} = (H_{in} - 1) * S_h + K_h - 2P  + P_{output}\\
&W_{out} = (W_{in} - 1) * S_w + K_w - 2P  + P_{output}
\end{aligned}$$
- `Wi,Hi`: 输入图像尺寸
- `Wo,Ho`: 输出图像尺寸
- `Kw,Kh`: 卷积核宽, 高
- `Sw,Sh`: 步长核宽, 高
-  `P`: 填充
- $P_{output}$: 用于解决输出尺寸歧义的附加填充

只有 **TensorFlow/Keras** 的 `Conv2DTranspose` 支持 `padding='same'`
$$H_{out}=H_{in}\times stride$$

---

## 3. nn.Linear - 全连接层

### 功能
实现线性变换 y = xA^T + b，用于特征转换。

### 基本语法
```python
nn.Linear(in_features, out_features, bias=True)
```

### 入参说明
| 参数 | 说明 | 类型 | 默认值 |
|-----|------|------|--------|
| `in_features` | 输入特征数 | int | 必需 |
| `out_features` | 输出特征数 | int | 必需 |
| `bias` | 是否使用偏置 | bool | True |

### 使用示例
```python
import torch
import torch.nn as nn

# 创建全连接层：输入512特征，输出10类
linear = nn.Linear(in_features=512, out_features=10)

# 输入张量：(batch_size, 512)
x = torch.randn(32, 512)
output = linear(x)
print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")  # (32, 10)
```

---

## 4. nn.Flatten - 展平层

### 功能
将高维张量展平为二维张量（保留batch维度）。

### 基本语法
```python
nn.Flatten(start_dim=1, end_dim=-1)
```

### 入参说明
| 参数 | 说明 | 类型 | 默认值 |
|-----|------|------|--------|
| `start_dim` | 开始展平的维度 | int | 1 |
| `end_dim` | 结束展平的维度 | int | -1 |

### 使用示例
```python
import torch
import torch.nn as nn

# 创建展平层
flatten = nn.Flatten(start_dim=1)

# 输入张量：(batch_size, 64, 7, 7)
x = torch.randn(32, 64, 7, 7)
output = flatten(x)
print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")  # (32, 3136) = 32 * 64 * 7 * 7
```

---

## 5. nn.Unflatten - 反展平层

### 功能
将展平的张量恢复为指定的多维形状。

### 基本语法
```python
nn.Unflatten(dim, unflattened_size)
```

### 入参说明
| 参数 | 说明 | 类型 | 默认值 |
|-----|------|------|--------|
| `dim` | 要展开的维度 | int | 必需 |
| `unflattened_size` | 目标形状 | tuple | 必需 |

### 使用示例
```python
import torch
import torch.nn as nn

# 创建反展平层：将dim=1展开为(64, 7, 7)
unflatten = nn.Unflatten(dim=1, unflattened_size=(64, 7, 7))

# 输入张量：(batch_size, 3136)
x = torch.randn(32, 3136)
output = unflatten(x)
print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")  # (32, 64, 7, 7)
```

---

## 6. nn.AdaptiveAvgPool2d - 自适应平均池化

### 功能
自适应地对输入进行平均池化，输出大小固定，无论输入大小如何。

### 基本语法
```python
nn.AdaptiveAvgPool2d(output_size)
```

### 入参说明
| 参数 | 说明 | 类型 | 默认值 |
|-----|------|------|--------|
| `output_size` | 输出空间大小 | int或tuple | 必需 |

### 使用示例
```python
import torch
import torch.nn as nn

# 创建自适应平均池化层：输出为(1, 1)
adaptive_avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))

# 输入张量：(batch_size, 64, H, W)，任意大小
x1 = torch.randn(32, 64, 224, 224)
output1 = adaptive_avg(x1)
print(f"输入1形状: {x1.shape}")
print(f"输出1形状: {output1.shape}")  # (32, 64, 1, 1)

x2 = torch.randn(32, 64, 128, 256)
output2 = adaptive_avg(x2)
print(f"输入2形状: {x2.shape}")
print(f"输出2形状: {output2.shape}")  # (32, 64, 1, 1)
```

---

## 7. nn.BatchNorm2d - 批量归一化

### 功能
对二维特征图进行批量归一化，加快训练速度，提高模型稳定性。

### 基本语法
```python
nn.BatchNorm2d(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
```

### 入参说明
| 参数 | 说明 | 类型 | 默认值 |
|-----|------|------|--------|
| `num_features` | 通道数 | int | 必需 |
| `eps` | 防止除以零的小值 | float | 1e-5 |
| `momentum` | 运行均值/方差的动量 | float | 0.1 |
| `affine` | 是否使用可学习的γ和β | bool | True |
| `track_running_stats` | 是否跟踪运行统计 | bool | True |

### 使用示例
```python
import torch
import torch.nn as nn

# 创建批量归一化层：64个通道
bn = nn.BatchNorm2d(num_features=64)

# 输入张量：(batch_size, 64, 224, 224)
x = torch.randn(32, 64, 224, 224)
output = bn(x)
print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")  # (32, 64, 224, 224)
print(f"均值接近0，方差接近1")
```

---

## 8. nn.MaxPool2d - 最大池化

### 功能
对特征图进行最大池化，降低维度，提取主要特征。

### 基本语法
```python
nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)
```

### 入参说明
| 参数 | 说明 | 类型 | 默认值 |
|-----|------|------|--------|
| `kernel_size` | 池化窗口大小 | int或tuple | 必需 |
| `stride` | 步长 | int或tuple | None(等于kernel_size) |
| `padding` | 填充大小 | int或tuple | 0 |
| `dilation` | 膨胀系数 | int或tuple | 1 |
| `ceil_mode` | 是否使用ceil模式 | bool | False |
| `return_indices` | 是否返回索引 | bool | False |

### 使用示例
```python
import torch
import torch.nn as nn

# 创建最大池化层：2x2窗口，步长为2
maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

# 输入张量：(batch_size, 64, 224, 224)
x = torch.randn(32, 64, 224, 224)
output = maxpool(x)
print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")  # (32, 64, 112, 112)
```

### 输出shape计算
$$
W_o = {floor}(\lfloor \frac{W_i - K_w + 2P}{S_w} \rfloor) + 1 \;\;\;\;\;\;\; H_o = floor(\lfloor \frac{H_i - K_h + 2P}{S_h} \rfloor) + 1
$$
- `Wi,Hi`: 输入图像尺寸
- `Wo,Ho`: 输出图像尺寸
- `Kw,Kh`: 卷积核宽, 高
- `Sw,Sh`: 步长核宽, 高
-  `P`: 填充

| padding   | 规则      |
| --------- | ------- |
| `'valid'` | ✅ 向下取整  |
| `'same'`  | ⚠️ 向上取整 |

---

## 9. nn.Embedding - 嵌入层

### 功能
将整数索引转换为稠密向量，常用于自然语言处理。

### 基本语法
```python
nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False)
```

### 入参说明
| 参数 | 说明 | 类型 | 默认值 |
|-----|------|------|--------|
| `num_embeddings` | 嵌入表大小（词汇量） | int | 必需 |
| `embedding_dim` | 嵌入向量维度 | int | 必需 |
| `padding_idx` | 填充索引 | int | None |
| `max_norm` | 最大范数 | float | None |
| `norm_type` | 范数计算类型 | float | 2 |
| `scale_grad_by_freq` | 是否按频率缩放梯度 | bool | False |
| `sparse` | 是否使用稀疏梯度 | bool | False |

### 使用示例
```python
import torch
import torch.nn as nn

# 创建嵌入层：词汇量10000，嵌入维度256
embedding = nn.Embedding(num_embeddings=10000, embedding_dim=256, padding_idx=0)

# 随机整数生成范围[0,10000]
# 32个样本，序列长度50
x = torch.randint(0, 10000, (32, 50))   
output = embedding(x)
print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")  # (32, 50, 256)
```

---

## 10. nn.RNN - 循环神经网络

### 功能
处理序列数据，捕捉时间依赖关系。

### 基本语法
```python
nn.RNN(input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0, bidirectional=False)
```

### 入参说明
| 参数 | 说明 | 类型 | 默认值 |
|-----|------|------|--------|
| `input_size` | 输入特征维度 | int | 必需 |
| `hidden_size` | 隐藏状态维度 | int | 必需 |
| `num_layers` | RNN层数 | int | 1 |
| `bias` | 是否使用偏置 | bool | True |
| `batch_first` | batch是否为第一维 | bool | False |
| `dropout` | Dropout比例 | float | 0 |
| `bidirectional` | 是否双向 | bool | False |

### 使用示例
```python
import torch
import torch.nn as nn

# 创建RNN：输入256维，隐藏层128维，2层
rnn = nn.RNN(input_size=256, hidden_size=128, num_layers=2, batch_first=True)

# 输入张量：(batch_size, sequence_length, input_size)
x = torch.randn(32, 50, 256)  # 32个样本，序列长度50，特征维度256
output, hidden = rnn(x)
print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")  # (32, 50, 128)
print(f"隐藏状态形状: {hidden.shape}")  # (2, 32, 128)
```

---

## 11. nn.LSTM - 长短期记忆网络

### 功能
改进的RNN，使用门机制解决长期依赖问题。

### 基本语法
```python
nn.LSTM(input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0, bidirectional=False)
```

### 入参说明
| 参数 | 说明 | 类型 | 默认值 |
|-----|------|------|--------|
| `input_size` | 输入特征维度 | int | 必需 |
| `hidden_size` | 隐藏状态维度 | int | 必需 |
| `num_layers` | LSTM层数 | int | 1 |
| `bias` | 是否使用偏置 | bool | True |
| `batch_first` | batch是否为第一维 | bool | False |
| `dropout` | Dropout比例 | float | 0 |
| `bidirectional` | 是否双向 | bool | False |

### 使用示例
```python
import torch
import torch.nn as nn

# 创建LSTM：输入256维，隐藏层128维，2层
lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=2, batch_first=True)

# 输入张量：(batch_size, sequence_length, input_size)
x = torch.randn(32, 50, 256)
output, (hidden, cell) = lstm(x)
print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")  # (32, 50, 128)
print(f"隐藏状态形状: {hidden.shape}")  # (2, 32, 128)
print(f"细胞状态形状: {cell.shape}")  # (2, 32, 128)
```

---

## 12. nn.GRU - 门控循环单元

### 功能
LSTM的简化版本，使用更少参数但性能相近。

### 基本语法
```python
nn.GRU(input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0, bidirectional=False)
```

### 入参说明
| 参数 | 说明 | 类型 | 默认值 |
|-----|------|------|--------|
| `input_size` | 输入特征维度 | int | 必需 |
| `hidden_size` | 隐藏状态维度 | int | 必需 |
| `num_layers` | GRU层数 | int | 1 |
| `bias` | 是否使用偏置 | bool | True |
| `batch_first` | batch是否为第一维 | bool | False |
| `dropout` | Dropout比例 | float | 0 |
| `bidirectional` | 是否双向 | bool | False |

### 使用示例
```python
import torch
import torch.nn as nn

# 创建GRU：输入256维，隐藏层128维，2层
gru = nn.GRU(input_size=256, hidden_size=128, num_layers=2, batch_first=True)

# 输入张量：(batch_size, sequence_length, input_size)
x = torch.randn(32, 50, 256)
output, hidden = gru(x)
print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")  # (32, 50, 128)
print(f"隐藏状态形状: {hidden.shape}")  # (2, 32, 128)
```

---

## 13. torch.squeeze
squeeze(input: Tensor, dim: Optional)
- **input** – the input tensor.
- **dim** – if given, the input will be squeezed only in the specified dimensions.

---
# 错题回顾
![[Pasted image 20251107222548.png]]![[Pasted image 20251107223058.png]]
![[Pasted image 20251108144250.png]]
![[Pasted image 20251108144410.png]]
![[Pasted image 20251108144453.png]]
![[Pasted image 20251108150642.png]]
![[Pasted image 20251108151626.png]]
![[Pasted image 20251108152616.png]]
![[Pasted image 20251108152704.png]]
![[Pasted image 20251111123640.png]]
![[Pasted image 20251108160349.png]]
![[Pasted image 20251108165059.png]]
![[Pasted image 20251108165536.png]]
![[Pasted image 20251108165731.png]]
![[9AB91B7133868A56BC1AF90EAEBC8C7E.png]]
![[Pasted image 20251108222742.png]]
![[Pasted image 20251108223210.png]]
![[Pasted image 20251108223655.png]]
![[Pasted image 20251108233234.png]]
![[Pasted image 20251108233416.png]]
![[Pasted image 20251108233820.png]]
![[Pasted image 20251108234145.png]]
![[Pasted image 20251108234251.png]]
![[Pasted image 20251108234604.png]]
![[Pasted image 20251109122829.png]]
![[Pasted image 20251109125427.png]]
![[Pasted image 20251109132820.png]]
![[Pasted image 20251109133630.png]]
![[Pasted image 20251109133810.png]]
![[Pasted image 20251109133903.png]]
![[Pasted image 20251109133938.png]]
![[Pasted image 20251109134000.png]]
![[Pasted image 20251109134016.png]]
![[Pasted image 20251109145437.png]]
![[Pasted image 20251109145839.png]]
![[Pasted image 20251109150059.png]]
![[Pasted image 20251109150324.png]]
![[Pasted image 20251109150338.png]]
![[Pasted image 20251110193545.png]]


# 大题预测
## 给出计算图，让求各个参数对h的偏导，考察chain rule得到应用
![[Pasted image 20251108155233.png]]

## 多维向量求导
![[Pasted image 20251108152105.png]]

## 序列模型求值
![[Pasted image 20251108215705.png]]

## TF-IDF
![[Pasted image 20251108225519.png]]
## Skip-gram
![[Pasted image 20251108231147.png]]
## CBOW
![[Pasted image 20251108231202.png]]
## Global Attention 计算
![[Pasted image 20251109123426.png]]
## Local Attention 计算
![[Pasted image 20251109123458.png]]
## Self-Attention计算
![[Pasted image 20251109125723.png]]
![[Pasted image 20251109125733.png]]
# ViTs shape
![[11666a21887c7c44b0df33707f5bc082_720.png]]
