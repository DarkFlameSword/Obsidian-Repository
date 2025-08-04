---
date: 2025-08-04
tags:
  - FIT5215
author:
  - Siyuan Liu
aliases:
  - note
---
# Saturated Activate Function
## `Sigmoid`
![[Pasted image 20250804154105.png]]
$$ \sigma(x) = \frac{1}{1 + e^{-x}} $$
==优:==

==缺:==
## `Tanh`
![[Pasted image 20250804154147.png]]
$$ \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} $$
# Unsaturated Activate Function
## `ReLU` 
![[Pasted image 20250804154218.png]]
$$ \mathrm{ReLU}(x) = \max(0, x) $$
## `Leaky Relu`
![[Pasted image 20250804154446.png]]
$$ \mathrm{LeakyReLU}(x) = \begin{cases} x, & x \geq 0 \ \alpha x, & x < 0 \end{cases} $$

## ELU
