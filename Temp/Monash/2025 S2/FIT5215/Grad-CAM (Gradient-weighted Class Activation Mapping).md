---
date: 2025-10-01
author:
  - Siyuan Liu
tags:
  - FIT5215
---
![[Pasted image 20251001094706.png]]
**计算过程:**
$$
L_{Grad-CAM}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)
$$
$$
\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{ij}^k}
$$
- $c:$ 目标类别
- $A^k:$ 第k个特征图（最后一层卷积的输出）
- $α_k^c:$ 第k个特征图对类别c的重要性权重
- $y^c:$ 类别c的得分（logit）
- $\frac{∂y^c}{∂A^k}:$ 类别得分关于特征图的梯度
- $Z:$ 特征图的像素总数（用于平均）
- $ReLU:$ 只保留正向影响

==简单来说：==
重要性权重 = 梯度的全局平均 
热力图 = ReLU(Σ 权重 × 特征图)

**前向传播获取激活**
```
# 输入图像
image = load_image("cat.jpg")  # (1, 3, 224, 224)

# 前向传播到最后一层卷积
conv_output = model.features(image)  # (1, 512, 7, 7)
# 512个特征图，每个7x7大小

# 继续到全连接层
logits = model.classifier(conv_output)  # (1, 1000)
predicted_class = logits.argmax()  # 假设预测为 "cat" (class 281)
```

**反向传播获取梯度**
```
# 对目标类别的logit求梯度
target_logit = logits[0, predicted_class]
target_logit.backward()

# 获取最后卷积层的梯度
gradients = conv_output.grad  # (1, 512, 7, 7)
# 每个特征图的每个位置都有梯度值
```

**计算权重（全局平均池化梯度）**
```
# 对每个特征图的梯度做全局平均池化
weights = torch.mean(gradients, dim=(2, 3))  # (1, 512)

# weights[k] 表示第k个特征图对预测的重要性
# 梯度大 → 该特征图对预测影响大
# 梯度小 → 该特征图对预测影响小
```

**加权求和特征图**
```
# 用权重加权特征图
cam = torch.zeros(conv_output.shape[2:])  # (7, 7)

for i in range(512):
    cam += weights[0, i] * conv_output[0, i]
    # 重要的特征图贡献更多

# cam 现在是一个7x7的热力图
```

**ReLU和归一化**
```
# 应用ReLU（只保留正向贡献）
cam = torch.relu(cam)

# 归一化到[0, 1]
cam = cam - cam.min()
cam = cam / cam.max()

# 上采样到原图大小
cam_upsampled = F.interpolate(
    cam.unsqueeze(0).unsqueeze(0),
    size=(224, 224),
    mode='bilinear'
)
```