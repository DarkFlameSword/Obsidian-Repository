---
date: 2025-10-01
author:
  - Siyuan Liu
tags:
  - FIT5215
---
**核心思想：**
当前值由过去的值线性组合加上随机误差项决定

**计算公式：**
$$Xₜ = c + φ₁Xₜ₋₁ + φ₂Xₜ₋₂ + ... + φₚXₜ₋ₚ + εₜ$$
- $X_t$：当前时刻的值
- $c$：常数项
- $φ₁, φ₂, ..., φₚ$：自回归系数
- $p$：阶数（使用过去多少个值）
- $ε_t$：白噪声误差项

**代码举例：**
```
import numpy as np
import matplotlib.pyplot as plt

# 生成 AR(1) 时间序列
np.random.seed(42)
n = 200
phi = 0.8  # 自回归系数
c = 10     # 常数项
noise = np.random.normal(0, 1, n)

# 初始化
X = np.zeros(n)
X[0] = c / (1 - phi)  # 理论均值

# 生成序列: X_t = c + φ * X_{t-1} + ε_t
for t in range(1, n):
    X[t] = c + phi * X[t-1] + noise[t]

# 可视化
plt.figure(figsize=(12, 4))
plt.plot(X)
plt.title('AR(1) 时间序列示例')
plt.xlabel('时间')
plt.ylabel('值')
plt.grid(True)
plt.show()

print(f"理论均值: {c/(1-phi):.2f}")
print(f"实际均值: {X.mean():.2f}")
```