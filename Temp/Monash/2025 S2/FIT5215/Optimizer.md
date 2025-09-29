---
date: 2025-08-18
author:
  - Siyuan Liu
tags:
  - FIT5215
aliases:
  - summary
---
# Stochastic Gradient Descent(SGD)

**核心思想:**
随机抽取部分样本，近似当前梯度

==**实际应用中的折中：小批量梯度下降 (Mini-batch GD)** 在实践中，我们通常不用纯粹的SGD（1个样本），也不用Batch GD（全部样本）。我们采用一个折中方案：每次随机取一小批数据（比如32、64或128个样本），用这个“mini-batch”来计算梯度并更新参数。这既利用了GPU并行计算的优势，又保持了更新的快速和随机性。现在人们通常说的SGD，大多指的都是这种Mini-batch SGD==
$$𝑊 = 𝑊 − 𝜂 \frac{\partial{l}}{\partial{W}}$$
$$b = b − 𝜂 \frac{\partial{l}}{\partial{b}}$$
$$\theta_{t+1} = \theta_t - \mu_t\nabla_{\theta}J(\theta_t)$$
- $𝜂$: learning rate
- $l$: **单个小批量 (mini-batch) 数据** 的损失
- $\nabla_{\theta}J(\theta_t):$ 梯度

**优点:**
这样每次更新都 **非常快**，只需随机抽取32,64,128,256...个样本

**缺点:**
更新方向比较“抖动”，不如全量梯度那么稳定

==举例==
![[Pasted image 20250822174832.png]]
![[Pasted image 20250822182635.png]]

|Optimizer|understand|
|---|---|
|**BGD**|像一个人走路前要看完整张地图（全量数据），走得稳，但慢|
|**SGD**|每次只看一个路标（一个样本），走得快，但路线有点抖动|
|**Mini-batch SGD**| 折中方案，比如每次用 32 个样本，既快又相对稳定|

---
## SGD with momentum

**采样一个 mini-batch**
$$\{(x_1, y_1), (x_2, y_2), \dots, (x_b, y_b)\}$$
这里 b 是 batch size。

 **计算梯度的 mini-batch 平均值**
$$g = \frac{1}{b}\sum_{i=1}^{b} \nabla_\theta \, l(f(x_i;\theta), y_i)$$
- $l(\cdot)$：损失函数
- $f(x_i;\theta)$：模型预测
- g：当前 batch 的平均梯度

**更新动量（velocity）**
$$v = \alpha v + (1-\alpha) g$$

- v：动量（类似“速度”）
- $\alpha \in [0,1))$：动量系数（常见 0.9）
- **解释**：新的速度是“历史速度的一部分 + 当前梯度的一部分”
    - 如果 $\alpha$ 很大（接近 1），说明更重视历史方向
    - 如果 $\alpha$ 较小，说明更重视当前梯度

**更新参数**
$$\theta = \theta - \eta v$$
- $\eta > 0$：学习率
- 参数的更新方向 = “动量方向”
- **好处**：不会被单个 batch 的噪声干扰太大，更新更平滑、更快收敛

==理解:==
- **普通 SGD**：小球每次只看当前位置的坡度，容易在谷底左右震荡。
- **SGD with momentum**：小球有“惯性”，会把之前的梯度方向也考虑进去，更快滚到谷底，而且震荡小
# AdaGrad
==理解:==
想象你在一个崎岖的山谷中试图走到谷底（损失函数的最小值）

- **标准梯度下降 (SGD)**：你每一步的大小（学习率）都是固定的。如果山谷在一个方向上非常陡峭，而在另一个方向上非常平缓，你固定的步长可能会在陡峭方向上来回震荡，同时在平缓方向上前进缓慢
    
- **AdaGrad (Adaptive Gradient Algorithm)**：你变得更“智能”了。你会记录下你走过的每一步历史
    - 对于那些你**经常移动**的方向（梯度一直很大的方向，即陡峭的坡），你会变得更加谨慎，**减小步长**，以防走过头
    - 对于那些你**很少移动**的方向（梯度一直很小的方向，即平缓的坡），你会变得更加大胆，**增**大步长，以加速探索

简而言之，AdaGrad 的核心思想是：**为每一个参数（Parameter）自适应地调整其学习率**。更新频繁的参数，其学习率会衰减得更快；更新稀疏的参数，其学习率会衰减得更慢

**计算步骤：**
**输入**
$$\begin{aligned}
& \eta > 0\\
& \epsilon > 0 (\less 10^{-6})\\
& \text{and initial model}\;\theta
\end{aligned}$$
- $\eta:$ 学习率 (learning rate)，控制参数更新的步长
- $\epsilon:$ 数值稳定项，通常设为 10⁻⁶，防止除零错误
- $\theta:$ 模型参数的初始值

**采样小批次数据**
$$\text{Sample a mini-batch}\; {(x₁, y₁), ..., (xᵦ, yᵦ)}$$
- 从训练集中随机采样一个小批次
- `b` 是批次大小 (batch size)
- `(xᵢ, yᵢ)` 是第 i 个样本的输入和标签
### 计算当前批次所有样本的平均梯度
$$g = \frac{1}{b} \sum_{i=1}^b \nabla_\theta l(f(x^i, \theta), y^i)$$
- `l(f(xᵢ, θ), yᵢ)`: 对第 i 个样本的损失函数
- `∇θl(...)`: 损失函数对参数 θ 的梯度
- `Σᵢ₌₁ᵇ`: 对批次中所有样本求和
- `(1/b)`: 计算平均梯度
### 累积平方梯度
$$\text{Accumulate the square gradient}: \gamma = \gamma + g\odot g$$
- `γ` (gamma): **累积平方梯度** 向量，初始为零向量
- `g⊙g`: 梯度的**逐元素平方** (element-wise square)
- `γ = γ + g⊙g`: 将当前梯度的平方累加到历史记录中
### 自适应参数更新
$$\theta = \theta - \frac{\eta}{\sqrt{\epsilon + \gamma}} \odot g$$
1. **`√γ`**: 累积平方梯度的平方根
2. **`ε + √γ`**: 加上数值稳定项
3. **`η/(ε + √γ)`**: **自适应学习率**，每个参数都有不同的学习率
4. **`η/(ε + √γ) ⊙ g`**: 逐元素相乘，得到最终的更新量
# RMSProp
# Adam
# Back Propagation in feed