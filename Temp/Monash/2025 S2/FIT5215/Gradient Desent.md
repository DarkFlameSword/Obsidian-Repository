---
date: 2025-08-22
author:
  - Siyuan Liu
tags:
  - FIT5215
aliases:
  - base
---
$θt+1​=θt​−η∇θ​J(θt​)$
- **θt​**：第 ttt 次迭代时的参数（模型的权重和偏置）。
- **θt+1\theta_{t+1}θt+1​**：更新后的参数，也就是下一次迭代用的参数。
- **η\etaη**：学习率（learning rate），一个超参数，控制每次更新的步长大小。
- **J(θt)J(\theta_t)J(θt​)**：损失函数（loss function），衡量当前参数下模型预测与真实标签的差距。
- **∇θJ(θt)\nabla_{\theta} J(\theta_t)∇θ​J(θt​)**：损失函数对参数的梯度（gradient），告诉我们“在当前位置，往哪个方向能最快增加损失”。