---
date: 2025-10-01
author:
  - Siyuan Liu
tags:
  - FIT5215
---
![[Pasted image 20251019145841.png]]
![[Pasted image 20251108222356.png]]
![[Pasted image 20251108221624.png]]

**运行过程：**

**Update gate 𝑧+** 
decides how much the unit updates its state
$$
z_t = sigmoid(W_z @ [h_{t-1}, x_t] + b_z)
$$
- $z_t ≈ 0:$ 保留旧状态 $h_{t-1}$，忽略新信息
- $z_t ≈ 1:$ 接受新信息 $g_t$，丢弃旧状态

**Reset gate $r_t$**
controls which parts of the state get used to compute the next target state
$$r_t = sigmoid(W_r @ [h_{t-1}, x_t] + b_r)$$
- $r_t ≈ 0$: 忽略过去状态，重新开始
- $r_t ≈ 1$: 完全使用过去状态

**Candidate State $g_t$**
$$\begin{aligned}
& \text{ResetHidden} = r_t * h_{t-1}\\
& g_t = tanh(W_g @ [ResetHidden, x_t] + b_g)
\end{aligned}$$
- $r_t ≈ 0$: $g_t$ 主要依赖 $x_t$（新起点）
- $r_t ≈ 1$: $g_t$ 结合 $h_{t-1}$ 和 $x_t$（延续）

**memory state $h_t$**
$$h_t = (1 - z_t) * h_prev + z_t * g_t$$
- $(1 - z_t) + z_t = 1$


==当 $z_t$ 和 $r_t$ 接近 1 时，GRU 退化为基本 RNN==