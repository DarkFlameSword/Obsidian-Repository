---
date: 2025-08-18
author:
  - Siyuan Liu
tags:
  - 随笔
aliases:
  - base
---
## **2. 上标与下标**

- 上标 `^`  
    `$x^2$` → x2x^2x2  
    多个字符需用 `{}`：`$x^{10}$` → x10x^{10}x10
    
- 下标 `_`  
    `$a_1$` → a1a_1a1​  
    多个字符需用 `{}`：`$a_{ij}$` → aija_{ij}aij​
    
- 上下标同时：`$x_i^2$` → xi2x_i^2xi2​
    

---

## **3. 分数**

- `\frac{分子}{分母}`  
    `$ \frac{a+b}{c+d} $` → a+bc+d\frac{a+b}{c+d}c+da+b​
    

---

## **4. 根号**

- 平方根：`\sqrt{...}`  
    `$ \sqrt{x} $` → x\sqrt{x}x​
    
- 任意次方根：`\sqrt[n]{...}`  
    `$ \sqrt[3]{x} $` → x3\sqrt[3]{x}3x​
    

---

## **5. 求和与积分**

- 求和：`\sum`  
$$ \sum_{i=1}^n i^2 $$ → ∑i=1ni2\sum_{i=1}^n i^2∑i=1n​i2
    
- 积分：`\int`  
    `$ \int_0^1 x^2 dx $` → ∫01x2dx\int_0^1 x^2 dx∫01​x2dx
    
- 极限：`\lim`  
    `$ \lim_{x \to 0} \frac{\sin x}{x} $` → lim⁡x→0sin⁡xx\lim_{x \to 0} \frac{\sin x}{x}limx→0​xsinx​
    

---

## **6. 括号与绝对值**

- 普通括号：`( ... )`
    
- 自动大小括号：`\left( ... \right)`  
    `$ \left( \frac{a}{b} \right) $` → (ab)\left(\frac{a}{b}\right)(ba​)
    
- 绝对值：`| ... |` 或 `\left| ... \right|`  
    `$ \left| x \right| $` → ∣x∣\left|x\right|∣x∣
    

---

## **7. 矢量、矩阵与行列式**

- 矢量：`\vec{v}` → v⃗\vec{v}v
    
- 单行矩阵：`\begin{matrix} ... \end{matrix}`
    
    `$ \begin{matrix} a & b \\ c & d \end{matrix} $`
    
    →
    
    abcd\begin{matrix} a & b \\ c & d \end{matrix}ac​bd​
- 带括号矩阵：
    
    `\begin{pmatrix} a & b \\ c & d \end{pmatrix}  % () \begin{bmatrix} a & b \\ c & d \end{bmatrix}  % [] \begin{Bmatrix} a & b \\ c & d \end{Bmatrix}  % {} \begin{vmatrix} a & b \\ c & d \end{vmatrix}  % | \begin{Vmatrix} a & b \\ c & d \end{Vmatrix}  % ||`
    

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

|命令|含义|示例|
|---|---|---|
|`\,`|1/6 个 `quad`，非常小|`$a\,b$` → a ba\,bab|
|`\:`|2/9 个 `quad`，稍大|`$a\:b$` → a ba\:bab|
|`\;`|5/18 个 `quad`，中等|`$a\;b$` → a  ba\;bab|
|`\quad`|大空格，相当于当前字体宽度的 1 倍|`$a\quad b$` → aba\quad bab|
|`\qquad`|特大空格，2 倍 `quad`|`$a\qquad b$` → aba\qquad bab|
## **12. 固定长度空格**

- `\hspace{长度}`：可以指定任意长度  
    示例：
    
    `$a\hspace{10pt}b$`
    
    → aba\hspace{10pt}bab
    
- 支持单位：`pt`（点），`cm`（厘米），`mm`（毫米），`em`（字体宽度），`ex`（x 高度）
## 13插入文本
使用 `\text{ }` 在公式中插入普通文本：

`$\text{anything}$`
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