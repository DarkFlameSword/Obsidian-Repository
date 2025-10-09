---
date: 2025-10-05
author:
  - Siyuan Liu
tags:
  - FIT5047
---
**定义：**
Classify objects based on the values of their explanatory attributes

**举例：**
![[Pasted image 20251009220608.png]]
# Avoiding Overfitting in Decision Trees
- Pre-pruning: Stop growing the tree if the goodness measure falls below a threshold

- Post-pruning: Grow the full tree, then prune (Use a validation dataset)

- Regularization: Add a complexity penalty to the performance measure
	E.g., Complexity = Number of nodes in the tree
