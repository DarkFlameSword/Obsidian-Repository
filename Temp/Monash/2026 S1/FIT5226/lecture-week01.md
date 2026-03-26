---
date: 2026-03-02
author:
  - Siyuan Liu
tags:
  - FIT5226
---
# MAS Complexity
### 1. 计算复杂性 (Computational Complexity)

这是最硬核、最偏数学的一面。在单智能体中，我们通常解一个马尔可夫决策过程 (MDP)。但在 MAS 中，当多个智能体只拥有局部视野并且需要合作时，问题就变成了**分散式部分可观察马尔可夫决策过程 (Dec-POMDP)**。

- **数学严谨性：** 求解有限视野的 Dec-POMDP 已被证明是 **NEXP-完全 (NEXP-complete)** 的。这意味着即使是非常小规模的多智能体协作问题，在数学上找到最优解的时间也会随着智能体数量和状态空间呈超指数级爆炸
- **博弈论复杂性：** 如果智能体是自私的（非纯合作），寻找纳什均衡 (Nash Equilibrium) 在计算上是 **PPAD-完全** 的，这本身就是一个极具挑战性的计算复杂度类

### 2. 交互与网络复杂性 (Interaction & Network Complexity)

MAS 的精髓在于“多”。随着智能体数量 $N$ 的增加，潜在的交互连接数呈 $O(N^2)$ 增长。

- **拓扑结构：** 智能体之间是如何连接的？是全连接、无标度网络 (Scale-free network) 还是小世界网络 (Small-world network)？网络拓扑直接决定了信息传播的效率和系统的鲁棒性
- **动态耦合：** 智能体 $A$ 的动作会改变环境，从而影响智能体 $B$ 的最优策略。这种相互依赖导致了策略空间的非平稳性 (Non-stationarity)，这是多智能体强化学习 (MARL) 面临的最大噩梦之一

### 3. 个体行为复杂性 (Agent-level Cognitive Complexity)

系统由什么样的个体组成？

- **反应式 (Reactive)：** 像蚂蚁一样，只根据当前局部刺激做出简单本能反应。个体极简，但依然能产生群体复杂性
- **慎思式 (Deliberative/BDI)：** 智能体拥有信念 (Beliefs)、愿望 (Desires) 和意图 (Intentions)。它们需要进行复杂的逻辑推理、规划甚至互相进行心智建模 (Theory of Mind：“我猜你是怎么猜我的”)

### 4. 环境复杂性 (Environmental Complexity)

智能体所处的舞台本身也是复杂的。

- 环境通常是**部分可观察的 (Partially Observable)**，也就是俗称的“战争迷雾”
- 环境是**随机的 (Stochastic)** 且**动态的 (Dynamic)**。除了自然环境的演变，其他智能体的行为也在不断改变环境的状态

### 5. 涌现与非线性复杂性 (Emergence & Non-linearity)

这是“集体行为 (Collective Behaviour)”领域最迷人的地方。

在复杂系统中，**整体不等于部分之和 ($1+1 > 2$)**。即使每个智能体遵循极其简单的局部规则（例如鱼群中的鱼只看周围几条鱼的距离和方向），系统在宏观层面上也会自发地“涌现”出高度复杂、有序甚至具有美感的全局模式（如鸟群的漩涡、蚁群的觅食网络）。这种非线性动态意味着我们很难通过纯粹的自上而下 (Top-down) 分析来预测系统的最终走向。

---

# Modelling Approaches
![[Pasted image 20260326170951.png]]