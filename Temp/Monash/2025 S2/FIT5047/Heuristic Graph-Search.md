---
date: 2025-08-10
author:
  - Siyuan Liu
tags:
  - FIT5047
aliases:
  - base
---
# Heuristic Function
$$f(n) = g(n) + h(n)$$
- **n**：代表图中的任意一个节点（状态）。
- **g(n)**：从**起点**到节点 **n** 的**实际代价**（已经付出的代价）。
- **h(n)**：从节点 **n** 到**终点**的**启发式估算代价**（Heuristic）。这是对未来代价的一个有根据的猜测，它不是真实值，而是算法的“直觉”。
- **f(n)**：从起点经过节点 **n** 到达终点的**估计总代价**。A* 算法会优先选择 `f(n)` 值最小的节点进行探索。
## Manhattan Distance
$$h(n) = |n.x - Goal.x| + |n.y - Goal.y|$$
**适用场景:** 当移动被限制在网格的四个方向（上、下、左、右）时
## Diagonal / Chebyshev Distance
$$h(n) = max( |n.x - Goal.x|, |n.y - Goal.y| )$$
**适用场景:** 当移动被限制在网格的八个方向（上、下、左、右和四个对角线）时
## Euclidean Distance
$$h(n) = sqrt( (n.x - goal.x)² + (n.y - goal.y)² )$$
**适用场景:** 当可以朝任意角度直线移动时
