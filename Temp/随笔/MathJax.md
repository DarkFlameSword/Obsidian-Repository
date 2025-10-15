---
date: 2025-08-18
author:
  - Siyuan Liu
tags:
  - 随笔
aliases:
  - base
---
# 上标   
- `$x^2$` → $x^2$
- `$x^{10}$` → $x^{10}$
# 头顶
- `\overline{}` →$\overline{a_3}\;$  适用于长表达式
- `\bar{}`→$\bar{a_3}\;$  主要用于单个字符
- `\vec{v}` → $\vec{v}$  向量
- `\overrightarrow{AB}` → $\overrightarrow{AB}$ 向量(适用于长表达式)
- `\hat{i}` → $\hat{i}$ 单位向量
- `\widehat{ab}` → $\widehat{ab}$  单位向量(适用于长表达式)
- `\tilde{x}` → x̃
- `\widetilde{ABC}` → $\widetilde{ABC}$
- `\dot{a}` → $\dot{a}$
- `\overbrace{(n−2)+(n−1)}` → $\overbrace{(n−2)+(n−1)}^2$

# 下标   
- `$a_1$` → $a_1$
- `$a_{ij}$` → $a_{ij}$
- `$x_i^2$` → $x_i^2$
- `$ \frac{a+b}{c+d} $` → $\frac{a+b}{c+d}$

# 下面

`\underbrace{(n−2)+(n−1)}` → $\underbrace{(n−2)+(n−1)}_2$
## 根号
- `$\sqrt{x}$` → $\sqrt{x}$
-  `$\sqrt[3]{x}$` → $\sqrt[3]{x}$

# 插入文本
`$\text{anything}$` → $\text{anything}$

## 括号

- 普通括号：`( ... )`
- 自适应大小括号：`$\left( \frac{a}{b} \right)$` → $\left( \frac{a}{b} \right)$

## 矩阵与行列式
- 单行矩阵：`$\begin{matrix} a & b \\ c & d \end{matrix}$` → $\begin{matrix} a & b \\ c & d \end{matrix}$
- 带括号矩阵：
`$\begin{pmatrix} -1 & 1 & -1 \\ 1 & 1 & -1\\ -1 & 1 & 1 \\ -1 & -1 & 2\end{pmatrix}$`  → $\begin{pmatrix} -1 & 1 & -1 \\ 1 & 1 & -1\\ -1 & 1 & 1 \\ -1 & -1 & 2\end{pmatrix}$
`$\begin{bmatrix} -1 & 1 & -1 \\ 1 & 1 & -1\\ -1 & 1 & 1 \\ -1 & -1 & 2\end{bmatrix}$` → $\begin{bmatrix} -1 & 1 & -1 \\ 1 & 1 & -1\\ -1 & 1 & 1 \\ -1 & -1 & 2\end{bmatrix}$

## 常用函数

- 三角函数：`\sin`, `\cos`, `\tan`, `\cot`, `\sec`, `\csc`  
    `$\sin x + \cos y$` → $\sin x + \cos y$
- 指数与对数：
    - `$e^x$` → $e^x$
    - `$\ln x$` → $\ln x$
    - `$\log_{10} x$` → $\log_{10} x$
- 导数：
    - `$\frac{d}{dx} f(x)$` → $\frac{d}{dx} f(x)$
    - `$f'(x)$` → $f'(x)$
- 偏导:
	- `$\frac{\partial y}{\partial x}$`→ $\frac{\partial y}{\partial x}$
- 极限
	- `\lim_{x \to \infty} f(x)` → $\lim_{x \to \infty} f(x)$
- 求和
	- `\sum_{i=1}^n i^2` →$\sum_{i=1}^n i^2$
- 积分
	-  `$\int_0^1 x^2 dx$` → $\int_0^1 x^2 dx$
	- `\iint_0^1xy dxdy` → $\iint_0^1xy \;dx dy$
- 自定义函数
	- `\operatorname{floor}(x)` → $\operatorname{floor}(x)$


## 空格命令

| 命令              | 含义                                                          | 示例                            |
| --------------- | ----------------------------------------------------------- | ----------------------------- |
| `\,`            | 1/6 个 `quad`，非常小                                            | `$a\,b$` → a ba\,bab          |
| `\:`            | 2/9 个 `quad`，稍大                                             | `$a\:b$` → a ba\:bab          |
| `\;`            | 5/18 个 `quad`，中等                                            | `$a\;b$` → a  ba\;bab         |
| `\quad`         | 大空格，相当于当前字体宽度的 1 倍                                          | `$a\quad b$` → aba\quad bab   |
| `\qquad`        | 特大空格，2 倍 `quad`                                             | `$a\qquad b$` → aba\qquad bab |
| `&`             | 矩阵空格                                                        |                               |
| `\hspace{10pt}` | 自定义空格(支持单位：`pt`（点），`cm`（厘米），`mm`（毫米），`em`（字体宽度），`ex`（x 高度）) | $A\hspace{10pt}B$             |

## 希腊字母

| 显示  | 语法       | 显示  | 语法     |
| --- | -------- | --- | ------ |
| γ   | \gamma   | δ   | \delta |
| ϵ   | \epsilon | ζ   | \zeta  |
| η   | \eta     | θ   | \theta |
| ι   | \iota    | κ   | \kappa |
| λ   | \lambda  | μ   | \mu    |
| ν   | \nu      | ξ   | \xi    |
| π   | \pi      | ρ   | \rho   |
| σ   | \sigma   | τ   | \tau   |
| υ   | \upsilon | ϕ   | \phi   |
| χ   | \chi     | ψ   | \psi   |
| ω   | \omega   |     |        |

# 常用符号

| 显示          | 语法          | 显示           | 语法              |
| ----------- | ----------- | ------------ | --------------- |
| ∞           | \infty      | ∪            | \cup            |
| ∩           | \cap        | ⊂            | \subset         |
| ⊆           | \subseteq   | ⊃            | \supset         |
| ∈           | \in         | ∉            | \notin          |
| ∅           | \varnothing | ∀            | \forall         |
| ∃           | \exists     | ¬            | \lnot           |
| ∇           | \nabla      | ∂            | \partial        |
| ⩾           | \ge         | ⩽            | \le             |
| >           | \gt         | <            | \lt             |
| ∧           | \land       | ∨            | \lor            |
| →           | \rightarrow | ↔            | \leftrightarrow |
| $\neq$      | \neq        | $\approx$    | \approx         |
| $a \cdot b$ | \cdot       | $a \times b$ | \times          |
| $a \div b$  | \div        | $\to$        | \to             |
| $\dots$     | \dots       | $\vdots$     | \vdots          |
| $\ddots$    | \ddots      | $\odot$      | \odot           |
| \           | \backslash  | $\sim$       | \sim            |
| ⊥           | \perp       | $\propto$    | \propto         |

# 包裹类命令
## 换行
```
$$
\begin{aligned}
a+b &= c \\
    &= d+e \\
    &= f
\end{aligned}
$$
```
- 在对齐位置前放 `&`
- 如果想行与行之间宽一点，则使用`\\[2ex]`
## 条件
$$ \alpha = 
\begin{cases} 
1 & \text{if sigmoid} \\ 
4 & \text{if tanh} \\ 
\sqrt{2} & \text{if ReLU} 
\end{cases} 
$$
```
$$ \alpha = 
\begin{cases} 
1 & \text{if sigmoid} \\ 
4 & \text{if tanh} \\ 
\sqrt{2} & \text{if ReLU} 
\end{cases} 
$$
```