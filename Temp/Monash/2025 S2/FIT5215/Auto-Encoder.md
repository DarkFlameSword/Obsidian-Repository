---
date: 2025-11-09
author:
  - Siyuan Liu
tags:
  - FIT5215
---
# Auto-Encoder
**核心思想**：
Learn good latent codes by minimizing the reconstruction error

自编码器是一种用于**数据压缩**和**特征学习**的无监督神经网络。你可以把它想象成一个“智能的有损压缩工具”，比如一个专家级的图像压缩程序。

它由两个主要部分组成：
1. **编码器 (Encoder)**：接收原始输入数据（如一张图片），并将其压缩成一个低维度的、包含核心信息的向量。这个向量被称为“潜在表示”或“编码”(latent representation / code)
2. **解码器 (Decoder)**：接收这个压缩后的编码，并尝试将其**重建**回原始的输入数据

模型的训练目标是让**重建后的输出**与**原始的输入**尽可能地相似。在这个过程中，编码器被迫学习如何提取数据中最重要的特征，以便解码器能够成功重建。

**优化公式：**
$$min_{\theta,\phi} E_{x\sim P} \;d(x, g_\phi(f_\theta(x)))$$
- $θ$: 编码器 $f_θ$ 的参数
- $ϕ$: 解码器 $g_ϕ$ 的参数
- $x$: 原始输入数据
- $P$: 数据分布
- $fθ(x)$: 编码器：将数据映射到**latent space**
- $gϕ(fθ(x))$: 解码器将潜在表示重构回原始空间
- $d(x,gϕ(fθ(x)))$: 重构误差距离函数
- $Ex∼P[⋅]$: 在整个数据分布上的期望
- $minθ,ϕ$: 同时优化两个参数集

**Perseudocode**
```
# 1. 定义模型结构
# 编码器将输入维度降低
encoder = EncoderNetwork(input_dim, latent_dim)
# 解码器将潜在维度恢复到原始维度
decoder = DecoderNetwork(latent_dim, input_dim)

# 2. 定义优化器和损失函数
# 损失函数衡量输入与输出的差异，均方误差是常用选择
loss_function = MeanSquaredError()
# 优化器用于更新网络权重
optimizer = Adam(parameters=[encoder.parameters(), decoder.parameters()], learning_rate=0.001)

# 3. 训练循环
for epoch in range(num_epochs):
    for batch_data in dataloader:
        # 获取原始输入数据
        original_input = batch_data

        # --- 前向传播 ---
        # 1. 编码：将输入压缩为潜在编码
        latent_code = encoder(original_input)
        # 2. 解码：从潜在编码重建输入
        reconstructed_output = decoder(latent_code)

        # --- 计算损失 ---
        # 损失是原始输入和重建输出之间的差异
        loss = loss_function(reconstructed_output, original_input)

        # --- 反向传播和优化 ---
        # 清空之前的梯度
        optimizer.zero_grad()
        # 计算梯度
        loss.backward()
        # 更新权重
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
```

---
# Sparse Auto-Encoder
**核心思想**：
在自编码器的基础上，对隐藏层（Hidden Layer）的神经元施加**稀疏性约束（Sparsity Constraint）**，迫使网络在任何给定时刻只有少量的神经元处于“激活”状态，而大多数神经元处于“抑制”（非激活）状态

**优化公式：**
$$min_{\theta,\phi} E_{x\sim P} \;[d(x, g_\phi(f_\theta(x)))]+ \lambda \Omega(z)$$
- **$x$**：输入数据（例如一张图片，一段文本的向量）
- **$\theta$ 和 $\phi$**：需要学习的网络参数（权重和偏置）
- **$f_\theta(x)$**：**编码器函数**。它将高维输入 $x$ 映射到低维（或稀疏）的隐藏层表示 $z$。即 $z = f_\theta(x)$
- **$g_\phi(z)$**：**解码器函数**。它将隐藏层表示 $z$ 映射回原始数据空间，得到重构后的数据 $\hat{x}$
- **$\Omega(z)$**：**正则化项（或惩罚项）**。这是对隐藏层表示 $z$（即 $f_\theta(x)$）施加的额外约束
    - 在稀疏自编码器（SAE的上下文中，$\Omega(z)$ 就是**稀疏惩罚**（例如 L1 范数 $||z||_1$），用于迫使 $z$ 中的大部分元素为 0
    - 在其他变体中，它可以是其他约束（例如收缩自编码器中的雅可比矩阵范数）
- **$\lambda$**：**超参数（正则化系数）**。它控制了“重构质量”和“约束强度”之间的权衡
    - $\lambda$ 越大，网络越重视满足约束（如更稀疏），但可能会牺牲重构的清晰度
    - $\lambda$ 越小，网络越重视还原数据，可能导致学到的特征不够本质或包含噪声

**作用：**
- **特征解耦（Disentanglement）**：稀疏性有助于学习到更加独立、含义更明确的特征。例如在人脸识别中，一个神经元可能专门负责识别“眼镜”，另一个专门负责“胡子”，而不是纠缠在一起
- **抗噪能力**：稀疏表示通常比稠密表示对噪声具有更强的鲁棒性
- **更高效的表示**：在高维空间中，稀疏表示往往能更高效地捕捉数据的内在结构