---
date: 2025-10-19
author:
  - Siyuan Liu
tags:
  - FIT5215
---
# Auto-Encoder
**核心思想**：
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

---
# GAN (Generative Adversarial Network)
**核心思想**：
GAN 是一种强大的**生成模型**，旨在创建全新的、与真实数据难以区分的数据（如逼真的名人照片、画作等）。它的灵感来源于一场“伪造者”与“鉴定师”之间的博弈。它也由两个主要部分组成，但它们是相互竞争的：

1. **生成器 (Generator)**：扮演**艺术伪造者**的角色。它接收一个随机噪声向量，使用Implicit Density Estimation尝试将其转换为看起来像真实数据的“伪造品”（例如，一张假图片）
2. **判别器 (Discriminator)**：扮演**“艺术鉴定师”**的角色。它接收一张图片（可能是真的，也可能是生成器伪造的），并判断这张图片的真伪

训练过程是一个“道高一尺，魔高一丈”的对抗游戏：

- 判别器努力学习如何区分真假数据
- 生成器努力学习如何“欺骗”判别器。 随着训练的进行，双方的能力都不断提升，最终生成器会变得非常出色，能够生成高度逼真的数据

**GAN的训练公式：**
$$Min_G\;Max_D\;J(G,D) = E_{x\sim P_{d}(x)}[logD(x)]+E_{z\sim P_z}[log(1-D(G(z))]$$

- $Ex∼P_d(x)$: 从真实数据分布 $P_d$ 中采样
- $x$: 真实数据（如真实图像）
- $D(x)$: 判别器输出（0-1 的概率）
- $logD(x)$: 判别器认为 x 是真实的概率的对数
- $Ez∼Pz$: 从噪声分布（如高斯分布）中采样
- $z$: 随机噪声向量
- $G(z)$: 生成器生成的假数据
- $D(G(z))$: 判别器认为生成数据是真实的概率
- $log(1−D(G(z)))$: 判别器认为生成数据是假数据的概率的对数


**Generator的训练公式：**
$$Min_\theta \;E_{z\sim P_{z}} [log(1-D(G(z))]$$
- Gradient descent to train the generator $D_\theta$

**Optimal generator：**
![[PixPin_2025-11-06_19-07-32 1.png]]

**Discriminator的训练公式：**
$$Max_\theta\;E_{x\sim P_{d}}[logD(x)]+E_{z\sim P_{z}}[log(1-D(G(z))]$$
- Gradient ascent to train discriminator $D_\theta$

**optimal discriminator：**
![[PixPin_2025-11-06_19-08-26.png]]


**Perseudocode**
```
# 1. 定义模型结构
# 生成器从噪声生成数据
generator = GeneratorNetwork(noise_dim, output_dim)
# 判别器判断数据真伪
discriminator = DiscriminatorNetwork(input_dim)

# 2. 定义优化器和损失函数
# 二元交叉熵损失，用于真/假二分类问题
loss_function = BinaryCrossEntropy()
# 需要为两个网络分别定义优化器
optimizer_G = Adam(generator.parameters(), learning_rate=0.0002)
optimizer_D = Adam(discriminator.parameters(), learning_rate=0.0002)

# 3. 训练循环
for epoch in range(num_epochs):
    for batch_data in dataloader:
        # 获取一批真实数据
        real_data = batch_data
        batch_size = real_data.size(0)

        # --- 步骤 1: 训练判别器 (Discriminator) ---
        # 目标：最大化 log(D(x)) + log(1 - D(G(z)))
        optimizer_D.zero_grad()

        # 1a: 用真实数据训练
        # 判别器对真实数据的预测结果
        prediction_real = discriminator(real_data)
        # 损失：我们希望判别器对真实数据输出 1 (真实)
        loss_real = loss_function(prediction_real, labels=torch.ones(batch_size))
        
        # 1b: 用伪造数据训练
        # 创建随机噪声作为生成器的输入
        noise = torch.randn(batch_size, noise_dim)
        # 生成伪造数据
        fake_data = generator(noise)
        # 判别器对伪造数据的预测结果
        # 使用 .detach() 防止梯度流回生成器
        prediction_fake = discriminator(fake_data.detach())
        # 损失：我们希望判别器对伪造数据输出 0 (伪造)
        loss_fake = loss_function(prediction_fake, labels=torch.zeros(batch_size))
        
        # 合并损失并更新判别器
        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizer_D.step()

        # --- 步骤 2: 训练生成器 (Generator) ---
        # 目标：最大化 log(D(G(z)))
        optimizer_G.zero_grad()
        
        # 我们需要重新让判别器对伪造数据进行判断（因为判别器刚刚更新过）
        prediction_G = discriminator(fake_data)
        # 损失：我们希望生成器能"欺骗"判别器，让判别器输出 1 (真实)
        loss_G = loss_function(prediction_G, labels=torch.ones(batch_size))
        
        # 更新生成器
        loss_G.backward()
        optimizer_G.step()

    print(f"Epoch {epoch+1}, Loss D: {loss_D.item()}, Loss G: {loss_G.item()}")
```



## The Nash equilibrium point
The solution of the minimax problem $Min_G\;Max_D\;J(G,D)$

Nash equilibrium point $(D^*,G^*)$ which satisfies
$$\begin{aligned}
&P_{g^*}=P_d\\
&D^*(x) = \frac{P_{d}(x)}{P_{d}(x)+P_{g^*}(x)}=0.5
\end{aligned}$$
- $p_d$: 真实数据的概率分布
- $P_{g^*}$: 最优生成器生成数据的概率分布
- $p_d(x)$: 数据 x 来自真实分布的概率
- $P_{d}(x)+P_{g^*}(x)$: 数据 x 来自任意来源的总概率

## Issues with GAN
- Mode collapsing problem
	**What is a "mode"?** In a dataset, a **"mode"** refers to a distinct cluster or type of data. For example, in the MNIST dataset of handwritten digits, the digits '0', '1', '2', etc., are all different modes. In a dataset of animal faces, "cats", "dogs", and "birds" would be different modes.

	**What is "Mode Collapse"?** Mode collapse happens when the **Generator gets lazy**. Instead of learning to create the full diversity of the real data (e.g., all 10 digits), it finds one or a few modes that are particularly easy to generate and are effective at fooling the Discriminator. The Generator then "collapses" on these few modes and produces very little variety in its output.
- Convergence is hard due to minimax formulation
- unrealistic generated images for complex datasets

---
# VAN (Variational Auto-Encoder)
**核心思想**：
VAE 是一种结合了自编码器和概率图模型思想的**生成模型**。它既能像 AE 一样学习数据的有效表示，又能像 GAN 一样生成新的数据。

它与标准 AE 的关键区别在于**编码器**和**损失函数**：

1. **概率编码器 (Probabilistic Encoder)**：VAE 的编码器使用Explicit Density Estimation。它会输出一个**概率分布**的参数，通常是高斯分布的**均值 (mean, `μ`)** 和**对数方差 (log-variance, `log(σ²)`**)。然后，我们从这个分布中**随机采样**一个点作为潜在编码 `z`
2. **双重损失函数 (Dual Loss Function)**：
    - **重建损失 (Reconstruction Loss)**：与 AE 相同，确保解码器能从 `z` 重建出原始输入
    - **KL 散度损失 (KL Divergence Loss)**：这是一个**正则化项**，它惩罚编码器输出的分布与标准正态分布（均值为0，方差为1）之间的差异。这个损失项迫使编码器学习一个**连续、规整**的潜在空间，使得我们可以在这个空间中随机采样来生成新的、有意义的数据

**Perseudocode**
```
# 1. 定义模型结构
# 编码器输出分布的均值和对数方差
encoder = EncoderNetwork(input_dim, latent_dim * 2) # *2 because of mean and log_var
decoder = DecoderNetwork(latent_dim, input_dim)

# "重参数化技巧" 函数，使得采样过程可微
def reparameterize(mean, log_var):
    std = torch.exp(0.5 * log_var) # 计算标准差
    epsilon = torch.randn_like(std) # 从标准正态分布中采样噪声
    return mean + epsilon * std # z = μ + ε * σ

# 2. 定义优化器和损失函数
optimizer = Adam(parameters=[encoder.parameters(), decoder.parameters()], learning_rate=0.001)

def loss_function(reconstructed_x, original_x, mean, log_var):
    # a. 重建损失 (例如，使用二元交叉熵或均方误差)
    reconstruction_loss = BinaryCrossEntropy(reconstructed_x, original_x, reduction='sum')
    
    # b. KL 散度损失
    # -0.5 * sum(1 + log(σ²) - μ² - σ²)
    kl_divergence_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    
    return reconstruction_loss + kl_divergence_loss

# 3. 训练循环
for epoch in range(num_epochs):
    for batch_data in dataloader:
        original_input = batch_data

        # --- 前向传播 ---
        # 1. 编码：获得分布的参数
        # [batch_size, latent_dim * 2] -> [batch_size, latent_dim] for mean and log_var
        mean, log_var = encoder(original_input).chunk(2, dim=1) 
        # 2. 采样：使用重参数化技巧从分布中采样 z
        z = reparameterize(mean, log_var)
        # 3. 解码：从 z 重建数据
        reconstructed_output = decoder(z)

        # --- 计算损失 ---
        # 损失包含重建损失和 KL 散度损失
        loss = loss_function(reconstructed_output, original_input, mean, log_var)

        # --- 反向传播和优化 ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
```

---
# Diffusion
**理解：**

修复一座被风沙侵蚀的沙堡
1. **侵蚀过程 (Forward Process - 加噪)**：你精心建造了一座完美的沙堡（**原始清晰图片**）。然后，你开始用摄像机记录风沙慢慢将其侵蚀的过程。你一帧一帧地拍摄，每一帧都比前一帧多了**一点点**沙子（**噪声**），直到最后，沙堡完全变成了一堆无法分辨的沙丘（**纯粹的随机噪声**）。这个“侵蚀”的过程是固定的、可预测的
2. **修复过程 (Reverse Process - 去噪)**：现在，你把录像**倒着播放**。你让一位技艺高超的艺术家（**AI 模型**）观看。艺术家的任务是，看着任何一帧被沙子覆盖的图像，准确地**预测出刚刚是哪些沙粒被吹了上来**。如果他能完美地预测出这些被添加的沙粒（噪声），他就能通过“吹走”这些沙粒，将图像恢复到前一帧更清晰的状态

扩散模型的核心，就是训练一个技艺精湛的“艺术家”模型，让它学会这个“修复”或“去噪”的过程

**扩散模型的两个关键过程**
1. 前向过程 (Forward Process)：加噪

这是一个固定的、无需学习的数学过程。

- **目标**：将一张清晰的图片，通过 `T` 个步骤（例如 `T=1000`），逐步地、缓慢地向其中添加高斯噪声
- **过程**：在每一步 `t`，我们都在第 `t-1` 步的图片上加入少量预设好的噪声，得到第 `t` 步的图片
    
    - $x_0 \xrightarrow{+\text{noise}} x_1 \xrightarrow{+\text{noise}} x_2 \rightarrow \dots \rightarrow x_T$
        
- **结果**：经过 `T` 步后，原始图片 $x_0$ 最终会变成一个与纯粹的高斯噪声无法区分的图像 $x_T$

**数学公式：**
$$x_t= x_{t-1}+ GaussianNoise$$
由上面的公式具象化为：
$$q(x_t|x_{t-1})=N(·|\mu_t,\epsilon_t)$$
- $q(x_t|x_{t-1})$: 在给定 $x_{t-1}$ 的条件下，$x_t$ 的概率分布
- $N(·|\mu_t,\epsilon_t)$: 表示 $x_t$ 服从一个**正态分布（高斯分布）**


$$\mu_t=\sqrt{1-\beta_t}\times x_{t-1}$$
- Mean
- $x_{t-1}$: 上一步的图像
- $\beta_t$: 一个非常小的、预先设定好的常数（例如 0.0001）。它会随着步骤 `t` 的增加而略微变大


$$\epsilon_t=\beta_tI$$
- Variance
- $\mathbf{I}$: 单位矩阵


$$x_t= \sqrt{1-\beta}\times x_{t-1}+\sqrt{\beta_t}\times \epsilon_{t-1}$$


$$x_t= \sqrt{\alpha_t}\times x_{0}+\sqrt{1-\alpha_t\epsilon_{0}}$$

---
#### 2. 反向过程 (Reverse Process)：去噪
去噪数学公式：
$$min_{\theta,\phi} E_{x\sim P} \big[\;E_{x'\sim N(x,\eta I)} [d(x, g_\phi(f_\theta(x')))]\;\big]$$
- $x\sim P$：真实图像P的数据分布$x$
- x′∼N(x,ηI)：添加高斯噪声后的图像$N(x,ηI)$的数据分布$x'$
- $f_\theta()$：encoder
- $g_\phi()$：decoder
- $d(x,g_\phi(f_\theta(x′)))$：去噪后图像与原始图像的距离


- **目标**：从一个纯噪声图像 $x_T$ 开始，通过 `T` 个步骤，逐步地将其恢复成一张清晰的图片 $x_0$
- **模型**：我们使用一个深度神经网络（通常是 U-Net 架构）来执行这个任务
- **训练任务**：在训练时，我们随机选择一个步骤 `t`，取出一张原始图片 $x_0$，并根据前向过程的公式直接生成其在第 `t` 步的噪声版本 $x_t$。然后，我们让模型看着 $x_t$ 和当前的步骤 `t`，去**预测在这一步被添加进去的噪声**
- **学习**：模型通过对比它预测的噪声和当时真实添加的噪声之间的差异（计算损失），来不断优化自己。最终，模型会变得极其擅长“看山是山，看沙是沙”——它能从混杂的图像中精准地识别出哪些是“沙”（噪声），哪些是“山”（原始信号）


 如何生成一张全新的图片？(推理/采样)

一旦模型训练完成，生成新图片的过程就变得非常神奇：

1. **第一步**：我们从一个标准正态分布中随机生成一个**纯噪声图像**（就像一块未经雕琢的画布，或一堆随机的沙丘）
2. **第二步**：我们将这个纯噪声图像（作为 $x_T$）和时间步 `T` 输入到我们训练好的去噪模型中
3. **第三步**：模型会预测出它认为应该从 $x_T$ 中移除的噪声。我们根据这个预测，从 $x_T$ 中减去一部分噪声，得到稍微清晰一点的图像 $x_{T-1}$
4. **第四步**：我们将 $x_{T-1}$ 和时间步 `T-1` 再次送入模型，模型又会预测出应该移除的噪声...
5. **循环往复**：我们重复这个过程 `T` 次，从 $x_T$ 一直走到 $x_0$。在这个过程中，一张清晰、连贯、全新的图片会奇迹般地从噪声中“浮现”出来

**对于文生图 (Text-to-Image)**，我们只是在每一步去噪时，额外给模型一个文本提示（例如“一只宇航员在骑马”）。模型在预测噪声时，不仅会考虑当前的噪声图像，还会考虑这个文本提示，从而引导整个去噪过程朝着我们想要的方向进行



## U-net
在扩散模型的**反向过程 (Reverse Process)** 中，U-Net 通过预测原始添加的噪声，来达到去噪的目标

U-Net 模型对噪声的**预测值**是$ϵ_θ(xₜ, t)$：
- **输入**：接收一个在时间步 `t` 的**噪声图像 `xₜ`** 和**当前的时间步 `t`** 本身
- **目标**：**预测**出当初为了从清晰图像 `x₀` 生成 `xₜ` 时，所加入的那个**原始高斯噪声 `ϵ`**
- **`θ`**: 代表了 U-Net 网络自身的可学习参数（权重和偏置）

