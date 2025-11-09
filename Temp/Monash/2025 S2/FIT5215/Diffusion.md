---
date: 2025-11-09
author:
  - Siyuan Liu
tags:
  - FIT5215
---
# Diffusion
## Forward Diffusion Process
Formulated as Markov chain with 𝑇 steps (usually large)

**At time 0:**
- Sample data point $𝑥_0$ from real data distribution $𝑞(𝑥)$ (i.e., from training data)

**At time t:**
- $x_t= x_{t-1}+ GaussianNoise$
- $𝑥_𝑡$ is now distributed according to $q(x_t|x_{t-1})=N(·|\mu_t,\epsilon_t)$ 
	- $\mu_t=\sqrt{1-\beta_t}\times x_{t-1}$
	- $Σ_𝑡= 𝛽_𝑡 𝑰$
	- $0 < 𝛽_𝑡< 1$

### How to sample $𝑥_𝑡$ from $N(x_t;\sqrt{1-\beta_t}\times x_{t-1},𝛽_𝑡 𝑰)$
use [[Reparameterization Trick]] to get 
$$x_t= \sqrt{1-\beta_t}\times x_{t-1}+\sqrt{\beta_t}\times \epsilon_{t-1}$$
### Is there an analytical form for $𝑥_𝑡$ directly from $𝑥_0$
![[Pasted image 20251109154007.png]]
![[Pasted image 20251109154038.png]]
## Backward Diffusion Process
Iteratively denoise from time step T to 0

**At time T:**
- Sample a random vector $𝑥_𝑇$ from $\mathbf{N}(0,\mathbf{I})$

**At time t:**
- $x_{t-1}=x_t - \text{amount of noise}$
- $x_{t-1}$ is now distributed according to $q(x_{t-1}|x_t)$
![[Pasted image 20251109154434.png]]
### U-net
在扩散模型的**反向过程 (Reverse Process)** 中，U-Net 通过预测原始添加的噪声，来达到去噪的目标

U-Net 模型对噪声的**预测值**是$ϵ_θ(xₜ, t)$：
- **输入**：接收一个在时间步 `t` 的**噪声图像 `xₜ`** 和**当前的时间步 `t`** 本身
- **目标**：**预测**出当初为了从清晰图像 `x₀` 生成 `xₜ` 时，所加入的那个**原始高斯噪声 `ϵ`**
- **`θ`**: 代表了 U-Net 网络自身的可学习参数（权重和偏置）

## How to train?
![[Pasted image 20251109155759.png]]
- **Fix a forward process (固定前向过程)**:
    - 定义了前向加噪过程。$x_t$ 是在 $t$ 时刻的加噪图像
    - 公式 $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$ 展示了如何直接从原始图像 $x_0$ 采样得到任意时刻 $t$ 的图像 $x_t$。这里 $\epsilon$ 是标准高斯噪声 $\mathcal{N}(0, I)$
    - $\beta_t$ 是预定义的噪声方差调度，$\alpha_t = 1 - \beta_t$，$\bar{\alpha}_t$ 是 $\alpha_s$ 的累乘

- **理想的反向过程**:
    - 我们希望从纯噪声 $x_T \sim \mathcal{N}(0, I)$ 开始，通过 $q(x_{t-1}|x_t)$ 一步步去噪生成新样本
    - **问题**: 真实的后验分布 $q(x_{t-1}|x_t)$ 是不可计算的（intractable），因为它需要知道整个数据分布

- **Solution (Ho et al.) (解决方案)**:
    - **关键洞察**: 如果我们知道原始图像 $x_0$，那么后验分布 $q(x_{t-1}|x_t, x_0)$ 就变得可计算了（tractable），并且它是一个高斯分布
    - 但是生成时我们没有 $x_0$
    - **方法**: 我们训练一个神经网络模型 $p_\theta(x_{t-1}|x_t)$ 来近似这个真实的后验分布。这类似于变分自编码器（VAE），通过最大化对数似然的变分下界（ELBO）来实现，这等价于最小化 $q(x_{t-1}|x_t, x_0)$ 和 $p_\theta(x_{t-1}|x_t)$ 之间的 KL 散度

- **How to parameterise and learn $p_\theta(x_{t-1}|x_t)$? (如何参数化和学习 $p_\theta$？)**
    - 假设 $p_\theta(x_{t-1}|x_t)$ 也是一个高斯分布，其均值为 $\mu_\theta(x_t, t)$，方差 $\Sigma(x_t, t)$ 设为固定值 $\sigma_t^2 I$（通常 $\sigma_t^2 = \beta_t$ 或 $\tilde{\beta}_t$）
    - 由于两个分布都是高斯分布且方差固定相同，最小化它们的 KL 散度就简化为最小化它们均值之间的均方误差（MSE）：$||\tilde{\mu}_t(x_t, x_0) - \mu_\theta(x_t, t)||^2$
    - 我们知道真实后验均值 $\tilde{\mu}_t$ 的解析形式（包含 $x_0$ 和噪声 $\epsilon$）
    - **核心思想**: 我们不直接预测均值 $\mu_\theta$，而是预测噪声 $\epsilon$。我们引入一个网络 $\epsilon_\theta(x_t, t)$ 来预测添加到图像中的噪声
    - 通过代换，最终的损失函数 $L_{t-1}$ 简化为：$||\epsilon - \epsilon_\theta(x_t, t)||^2$。即：**训练神经网络去预测（并减去）图像中的噪声**


**Pseudocode:**
![[Pasted image 20251109155826.png]]
Algorithm 1: Training (训练算法)

1. 从数据集中随机采样一张干净的图片 $x_0$。
    
2. 随机采样一个时间步 $t$（从 1 到 $T$）。
    
3. 随机采样一个高斯噪声 $\epsilon \sim \mathcal{N}(0, \mathbf{I})$。
    
4. **核心步骤**: 计算梯度下降。
    
    - 模型的输入是：加噪后的图片 $\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$（即 $x_t$） 和时间步 $t$。
        
    - 模型的目标是：预测出刚才加进去的那个噪声 $\epsilon$。
        
    - 损失函数是预测噪声和真实噪声之间的均方误差（MSE）。
        

Algorithm 2: Sampling (采样算法)

这是用训练好的模型生成新图片的过程：

1. 从标准高斯分布采样一个纯噪声 $x_T \sim \mathcal{N}(0, \mathbf{I})$。
    
2. 从 $t = T$ 到 $1$ 进行循环（倒序去噪）：
    
    - 采样一个额外的噪声 $z$（如果 $t > 1$），用于增加随机性（Langevin 动力学）。
        
    - **核心步骤**: 更新 $x_{t-1}$。公式里的红色框部分 $\frac{1}{\sqrt{\alpha_t}} (x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t))$ 就是模型预测的去噪后的均值 $\mu_\theta(x_t, t)$。
        
    - 简单理解就是：**当前图像减去模型预测的噪声（按一定比例），再加上一点点随机扰动**。
        
3. 循环结束，返回最终生成的图片 $x_0$。


---
# Noise Scheduler
## Linear schedule
$$\bar{\alpha}_t=\Pi^t_{s=0}(1-\beta_s)$$
- $\beta_s$ is constant
![[Pasted image 20251109155207.png]]

---
## Cosine schedule
$$\bar{\alpha}=\frac{f(t)}{f(0)}$$
- $f(t)=cos(\frac{\frac{t}{T}+S}{1+S}\times \frac{\pi}{2})^2$
![[Pasted image 20251109155216.png]]

---
# Latent Diffusion Model
- Perceptual image compression via Encoder $𝜀$ and Decoder $𝒟$
- Latent diffusion process with denoising U-Net $𝜖_𝜃$
- Conditioning mechanism so that we can generate images by a prompt

![[Pasted image 20251109155616.png]]