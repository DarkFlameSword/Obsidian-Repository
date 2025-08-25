---
date: 2025-08-18
author:
  - Siyuan Liu
tags:
  - 随笔
aliases:
  - base
---
# 1. 上标   
- `$x^2$` → $x^2$
- `$x^{10}$` → $x^{10}$
    
# 2. 下标   
- `$a_1$` → $a_1$
- `$a_{ij}$` → $a_{ij}$
- `$x_i^2$` → $x_i^2$
- `$ \frac{a+b}{c+d} $` → $\frac{a+b}{c+d}$
## **3. 根号**
- `$\sqrt{x}$` → $\sqrt{x}$
-  `$\sqrt[3]{x}$` → $\sqrt[3]{x}$

# 4. 插入文本
`$\text{anything}$` → $\text{anything}$
## **5. 求和与积分**

- `\sum_{i=1}^n i^2` →$\sum_{i=1}^n i^2$
- `$\int_0^1 x^2 dx$` → $\int_0^1 x^2 dx$
- `$\lim_{x \to 0} \frac{\sin x}{x}$` → $\lim_{x \to 0} \frac{\sin x}{x}$
## **6. 括号与绝对值**

- 普通括号：`( ... )`
- 自动大小括号：`$\left( \frac{a}{b} \right)$` → $\left( \frac{a}{b} \right)$
- 绝对值：`$\left| x \right|$` → $\left| x \right|$
## **7. 矢量、矩阵与行列式**

- 矢量：`\vec{v}` → $\vec{v}$
    
- 单行矩阵：`$\begin{matrix} a & b \\ c & d \end{matrix}$` → $\begin{matrix} a & b \\ c & d \end{matrix}$
- 带括号矩阵：
`$\begin{pmatrix} -1 & 1 & -1 \\ 1 & 1 & -1\\ -1 & 1 & 1 \\ -1 & -1 & 2\end{pmatrix}$`  → $\begin{pmatrix} -1 & 1 & -1 \\ 1 & 1 & -1\\ -1 & 1 & 1 \\ -1 & -1 & 2\end{pmatrix}$
`$\begin{bmatrix} -1 & 1 & -1 \\ 1 & 1 & -1\\ -1 & 1 & 1 \\ -1 & -1 & 2\end{bmatrix}$` → $\begin{bmatrix} -1 & 1 & -1 \\ 1 & 1 & -1\\ -1 & 1 & 1 \\ -1 & -1 & 2\end{bmatrix}$
## **8. 常用函数**

- 三角函数：`\sin`, `\cos`, `\tan`, `\cot`, `\sec`, `\csc`  
    `$\sin x + \cos y$` → $\sin x + \cos y$
- 指数与对数：
    - `$e^x$` → $e^x$
    - `$\ln x$` → $\ln x$
    - `$\log_{10} x$` → $\log_{10} x$
- 极限与导数：
    - `$\frac{d}{dx} f(x)$` → $\frac{d}{dx} f(x)$
    - `$f'(x)$` → $f'(x)$
## **9. 逻辑与集合符号**
- `$\neq$` → $\neq$
- `$\approx$` → $\approx$
- `$\le$` → $\le$
- `$\ge$` → $\ge$
- `$\in$` →  $\in$
- `$\notin$` → $\notin$
- `$\subset$` → $\subset$
- `$A \subseteq B$` → $A \subseteq B$
- `$\cup$` → $\cup$
- `$\cap$` → $\cap$
## **10. 其他符号**
- `$a \cdot b$` → $a \cdot b$
- `$a \times b$` → $a \times b$
- `$a \div b$` → $a \div b$
- `$\lim_{x \to \infty} f(x)$` → $\lim_{x \to \infty} f(x)$
- `$\dots$` → $1,2,3,\dots, n$
- `$\vdots$` → $1,2,3,\vdots, n$
- `$\ddots$` → $1,2,3,\ddots, n$
## **11. 小空格命令**

| 命令              | 含义                                                          | 示例                            |
| --------------- | ----------------------------------------------------------- | ----------------------------- |
| `\,`            | 1/6 个 `quad`，非常小                                            | `$a\,b$` → a ba\,bab          |
| `\:`            | 2/9 个 `quad`，稍大                                             | `$a\:b$` → a ba\:bab          |
| `\;`            | 5/18 个 `quad`，中等                                            | `$a\;b$` → a  ba\;bab         |
| `\quad`         | 大空格，相当于当前字体宽度的 1 倍                                          | `$a\quad b$` → aba\quad bab   |
| `\qquad`        | 特大空格，2 倍 `quad`                                             | `$a\qquad b$` → aba\qquad bab |
| `&`             | 矩阵空格                                                        |                               |
| `\hspace{10pt}` | 自定义空格(支持单位：`pt`（点），`cm`（厘米），`mm`（毫米），`em`（字体宽度），`ex`（x 高度）) | $A\hspace{10pt}B$             |

## 14

| Desired Output    | MathJax Syntax        | Rendered Result       |
| ----------------- | --------------------- | --------------------- |
| Vector (short)    | `\vec{v}`             | $\vec{v}$             |
| Vector (long)     | `\overrightarrow{AB}` | $\overrightarrow{AB}$ |
| Unit Vector (hat) | `\hat{i}`             | $\hat{i}$             |
| Bold Letter       | `\mathbf{x}`          | $\mathbf{x}$          |
## 15

