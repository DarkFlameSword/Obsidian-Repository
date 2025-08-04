---
date: 2025-07-29
author:
  - Siyuan Liu
tags:
  - FIT5047
aliases:
  - note
---
# Search Strategies

## Tentative Strategy 试探性策略
### Uninformed 盲目搜索
Backtracking 回溯
理解: 专注于**next step** ,如果**next step** 失败则回溯到上一个状态, 然后选择另一条路

Tree- and Graph search
理解: 从起点出发，沿着所有可能的通路走（比如广度优先、深度优先），直到找到出口。你不提前判断哪条路可能更好，只是机械地遍历所有路径
### informed 启发式搜索
#### Greedy best-first search 贪婪最佳优先搜索
理解: 每一步都选择“离goal最近”的下一个state，不考虑已走过的距离，只看剩余距离最短
#### A
理解: 每次扩展“估价函数f(n)”最小的节点，f(n)=g(n)+h(n)。
- g(n)：从起点到当前节点n的实际代价（已知）。
- h(n)：从n到目标的启发式估价（用来预测，通常用启发函数估算）
#### A*
理解: A* 算法是A算法的一个特例, A* 对启发函数h(n)有严格要求（必须可采纳）
- h(n)必须是“可采纳的/乐观的/低估的”（admissible）：即h(n)永远不能高估从n到目标的真实最小代价。
- 如果h(n)满足可采纳性，则A*算法保证找到一条**最优路径**。

## Irrevocable Strategy 不可逆策略
### Informed
#### Hill climbing
理解: 不断微调频率，只要信号变强就继续调，直到信号无法再增强为止（但可能只是“局部最优”）
#### Local beam search 局部束搜索
理解: 初始挑选5个人，每轮只保留表现最好的5人，淘汰差的，再各自扩展新候选人，继续选最优
#### Simulated annealing 模拟退火
理解: 爬山法基础上，偶尔允许“走下坡路”，以一定概率接受更差的解，有助于跳出局部最优点
#### Genetic algorithms 遗传算法
理解: 一群解（个体）交叉、变异，优胜劣汰。比如用遗传算法优化神经网络结构，每代都会产生新一批组合，保留表现最好的

## Adversarial search algorithms 对抗性搜索算法

### Optimal decisions
### Minimax

### α-β pruning