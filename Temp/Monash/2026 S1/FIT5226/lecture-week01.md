---
date: 2026-03-02
author:
  - Siyuan Liu
tags:
  - FIT5226
---
# Numpy
在 NumPy 中，当你对某一维使用布尔数组进行索引时，它会只保留对应位置为 `True` 的数据
```python
a = np.array([[3, 2, 5], [1, 4, 6]])

a[1, :] > a[0, :] # array([False,  True,  True])

a[:, a[1, :] > a[0, :]] # array([[2, 5],[4, 6]])
```