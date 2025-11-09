---
date: 2025-11-09
author:
  - Siyuan Liu
tags:
  - FIT5215
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
- $D(x) = P(y=1|x)$: probability x is true data.
- $1-D(x) = P(y=-1|x)$: probability x is fake data.

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
# DCGAN
![[Pasted image 20251109152441.png]]
- Replace any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator)
- Use batch normalization in both the generator and the discriminator
- Remove fully connected hidden layers for deeper architectures
- Use ReLU activation in generator for all layers except for the output which uses tanh
- Use LeakyReLU activation in the discriminator for all layers