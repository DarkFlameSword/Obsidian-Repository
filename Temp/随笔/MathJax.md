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

- 矢量：`\vec{v}` → v⃗\vec{v}v
    
- 单行矩阵：`$\begin{matrix} a & b \\ c & d \end{matrix}$` → $\begin{matrix} a & b \\ c & d \end{matrix}$
- 带括号矩阵：
    
    `$\begin{pmatrix} a & b \\ c & d \end{pmatrix}  % () \begin{bmatrix} a & b \\ c & d \end{bmatrix}  % [] \begin{Bmatrix} a & b \\ c & d \end{Bmatrix}  % {} \begin{vmatrix} a & b \\ c & d \end{vmatrix}  % | \begin{Vmatrix} a & b \\ c & d \end{Vmatrix}  % ||$` - $\begin{pmatrix} a & b \\ c & d \end{pmatrix}  % () \begin{bmatrix} a & b \\ c & d \end{bmatrix}  % [] \begin{Bmatrix} a & b \\ c & d \end{Bmatrix}  % {} \begin{vmatrix} a & b \\ c & d \end{vmatrix}  % | \begin{Vmatrix} a & b \\ c & d \end{Vmatrix}  % ||$
    

---

## **8. 常用函数**

- 三角函数：`\sin`, `\cos`, `\tan`, `\cot`, `\sec`, `\csc`  
    `$ \sin x + \cos y $` → sin⁡x+cos⁡y\sin x + \cos ysinx+cosy
    
- 指数与对数：
    
    - `$ e^x $` → exe^xex
        
    - `$ \ln x $` → ln⁡x\ln xlnx
        
    - `$ \log_{10} x $` → log⁡10x\log_{10} xlog10​x
        
- 极限与导数：
    
    - `$ \frac{d}{dx} f(x) $` → ddxf(x)\frac{d}{dx} f(x)dxd​f(x)
        
    - `$ f'(x) $` → f′(x)f'(x)f′(x)
        

---

## **9. 逻辑与集合符号**

- 等号与不等号：`=, \neq, \approx, \le, \ge`
    
- 集合：`\in, \notin, \subset, \subseteq, \cup, \cap`  
    `$ A \subseteq B $` → A⊆BA \subseteq BA⊆B
    

---

## **10. 其他符号**

- 分隔符：`\cdot, \times, \div`  
    `$ a \cdot b, a \times b $` → a⋅b,a×ba \cdot b, a \times ba⋅b,a×b
    
- 无穷：`\infty`  
    `$ \lim_{x \to \infty} f(x) $` → lim⁡x→∞f(x)\lim_{x \to \infty} f(x)limx→∞​f(x)
    
- 省略号：
    
    - 水平省略：`\dots` → 1,2,3,…,n1,2,3,\dots,n1,2,3,…,n
        
    - 垂直省略：`\vdots`
        
    - 对角省略：`\ddots`
        

---

如果你想，我可以帮你做一张 **MathJax 速查表 PDF/图片**，把这些常用语法都放在一页里，方便随时查。

你希望我做吗？

##### 您说：

空格如何表示

###### ChatGPT 说：

在 **MathJax/LaTeX** 中，空格的处理和普通文字不同，因为公式环境会自动忽略多余的空格。你可以用一些特殊命令来控制空格。常用方法如下：

---

## **11. 小空格命令**

| 命令       | 含义                 | 示例                            |
| -------- | ------------------ | ----------------------------- |
| `\,`     | 1/6 个 `quad`，非常小   | `$a\,b$` → a ba\,bab          |
| `\:`     | 2/9 个 `quad`，稍大    | `$a\:b$` → a ba\:bab          |
| `\;`     | 5/18 个 `quad`，中等   | `$a\;b$` → a  ba\;bab         |
| `\quad`  | 大空格，相当于当前字体宽度的 1 倍 | `$a\quad b$` → aba\quad bab   |
| `\qquad` | 特大空格，2 倍 `quad`    | `$a\qquad b$` → aba\qquad bab |
| `&`      | 矩阵空格               |                               |
## **12. 固定长度空格**

- `\hspace{长度}`：可以指定任意长度  
    示例：
    
    `$a\hspace{10pt}b$`
    
    → aba\hspace{10pt}bab
    
- 支持单位：`pt`（点），`cm`（厘米），`mm`（毫米），`em`（字体宽度），`ex`（x 高度）

## 14

| Desired Output    | MathJax Syntax        | Rendered Result       |
| ----------------- | --------------------- | --------------------- |
| Vector (short)    | `\vec{v}`             | $\vec{v}$             |
| Vector (long)     | `\overrightarrow{AB}` | $\overrightarrow{AB}$ |
| Unit Vector (hat) | `\hat{i}`             | $\hat{i}$             |
| Bold Letter       | `\mathbf{x}`          | $\mathbf{x}$          |
## 15
| ≤   | `\le`       | 小于等于 |
| --- | ----------- | ---- |
| ≥   | `\ge`       | 大于等于 |
| ≠   | `\ne`       | 不等于  |
| <   | `<` 或 `\lt` | 小于   |
| >   | `>` 或 `\gt` | 大于   |
| ≈   | `\approx`   | 约等于  |
| ≡   | `\equiv`    | 恒等于  |