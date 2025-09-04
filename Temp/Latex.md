---
date: 2025-09-04
author:
  - Siyuan Liu
tags:
  - 随笔
aliases:
  - summary
---
## 1. 文档结构

### 文档类与基本结构

latex

```latex
\documentclass[options]{class}  % 文档类声明
\begin{document}...\end{document}  % 文档主体
\usepackage{package}  % 导入宏包
```

### 章节命令

latex

```latex
\part{标题}
\chapter{标题}  % 仅在 book/report 类中
\section{标题}
\subsection{标题}
\subsubsection{标题}
\paragraph{标题}
\subparagraph{标题}
```

### 标题页与摘要

latex

```latex
\title{标题}
\author{作者}
\date{日期}
\maketitle  % 生成标题页
\begin{abstract}...\end{abstract}  % 摘要
\tableofcontents  % 目录
```

## 2. 文本格式化

### 字体样式

latex

```latex
\textbf{粗体}  % 粗体
\textit{斜体}  % 斜体
\texttt{等宽}  % 等宽字体
\textsc{小型大写}  % 小型大写
\textrm{罗马体}  % 罗马字体
\textsf{无衬线}  % 无衬线字体
\underline{下划线}  % 下划线
\emph{强调}  % 强调文本
```

### 字体大小

latex

```latex
\tiny  % 最小
\scriptsize
\footnotesize
\small
\normalsize  % 正常
\large
\Large
\LARGE
\huge
\Huge  % 最大
```

### 文本对齐

latex

```latex
\begin{center}...\end{center}  % 居中
\begin{flushleft}...\end{flushleft}  % 左对齐
\begin{flushright}...\end{flushright}  % 右对齐
\centering  % 段落居中
\raggedright  % 左对齐
\raggedleft  % 右对齐
```

## 3. 数学模式

### 数学环境

latex

```latex
$...$  % 行内数学模式
\(...\)  % 行内数学模式（推荐）
$$...$$  % 独立数学模式
\[...\]  % 独立数学模式（推荐）
\begin{equation}...\end{equation}  % 带编号的公式
\begin{equation*}...\end{equation*}  % 不带编号的公式
\begin{align}...\end{align}  % 多行对齐公式
```

### 常用数学符号

latex

```latex
% 上下标
x^2  % 上标
x_i  % 下标
x^{2n}  % 多字符上标
x_{ij}  % 多字符下标

% 分数与根号
\frac{a}{b}  % 分数
\sqrt{x}  % 平方根
\sqrt[n]{x}  % n次根

% 求和与积分
\sum_{i=1}^{n}  % 求和
\prod_{i=1}^{n}  % 求积
\int_{a}^{b}  % 积分
\oint  % 环积分
\lim_{x \to \infty}  % 极限
```

### 希腊字母

latex

```latex
\alpha  \beta  \gamma  \delta  % 小写
\epsilon  \zeta  \eta  \theta
\lambda  \mu  \nu  \xi
\pi  \rho  \sigma  \tau
\phi  \chi  \psi  \omega

\Gamma  \Delta  \Theta  \Lambda  % 大写
\Xi  \Pi  \Sigma  \Phi  \Psi  \Omega
```

### 数学运算符

latex

```latex
\pm  \mp  \times  \div  % 基本运算
\cdot  \ast  \star  \circ
\leq  \geq  \neq  \approx  % 比较
\equiv  \sim  \propto
\in  \notin  \subset  \supset  % 集合
\cup  \cap  \emptyset
\forall  \exists  \nexists  % 逻辑
\land  \lor  \neg  \implies
```

## 4. 列表环境

### 无序列表

latex

```latex
\begin{itemize}
  \item 项目一
  \item 项目二
\end{itemize}
```

### 有序列表

latex

```latex
\begin{enumerate}
  \item 第一项
  \item 第二项
\end{enumerate}
```

### 描述列表

latex

```latex
\begin{description}
  \item[术语] 描述内容
  \item[概念] 解释说明
\end{description}
```

## 5. 表格

### 基本表格

latex

```latex
\begin{tabular}{|l|c|r|}  % l左对齐 c居中 r右对齐
  \hline
  左对齐 & 居中 & 右对齐 \\
  \hline
  内容1 & 内容2 & 内容3 \\
  \hline
\end{tabular}
```

### 浮动表格

latex

```latex
\begin{table}[htbp]
  \centering
  \caption{表格标题}
  \begin{tabular}{cc}
    列1 & 列2 \\
    \hline
    数据1 & 数据2 \\
  \end{tabular}
  \label{tab:example}
\end{table}
```

### 表格命令

latex

```latex
\multicolumn{n}{格式}{内容}  % 合并列
\multirow{n}{宽度}{内容}  % 合并行（需要 multirow 宏包）
\cline{i-j}  % 部分横线
\hline  % 横线
```

## 6. 图片插入

latex

```latex
\usepackage{graphicx}  % 在导言区

\includegraphics[width=0.5\textwidth]{图片文件名}

% 浮动图片
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.8\textwidth]{filename}
  \caption{图片说明}
  \label{fig:example}
\end{figure}
```

### 图片选项

latex

```latex
width=5cm  % 指定宽度
height=3cm  % 指定高度
scale=0.5  % 缩放比例
angle=90  % 旋转角度
```

## 7. 引用与交叉引用

latex

```latex
\label{标签名}  % 设置标签
\ref{标签名}  % 引用编号
\pageref{标签名}  % 引用页码
\cite{文献标签}  % 引用参考文献
\footnote{脚注内容}  % 脚注
```

## 8. 特殊字符与空格

### 特殊字符

latex

```latex
\%  \$  \&  \#  \_  % 转义字符
\{  \}  \textbackslash  % 大括号与反斜杠
\S  \P  \dag  \ddag  % 特殊符号
\copyright  \pounds  \euro  % 版权、货币符号
```

### 空格控制

latex

```latex
~  % 不间断空格
\,  % 小空格
\:  % 中等空格
\;  % 大空格
\quad  % 1em空格
\qquad  % 2em空格
\\  % 换行
\newline  % 换行
\par  % 新段落
\vspace{长度}  % 垂直空间
\hspace{长度}  % 水平空间
```

## 9. 环境定义

### 定理环境

latex

```latex
\newtheorem{theorem}{定理}
\newtheorem{lemma}{引理}
\newtheorem{proposition}{命题}
\newtheorem{corollary}{推论}
\newtheorem{definition}{定义}
\newtheorem{example}{例}
\newtheorem{remark}{注}
```

### 代码环境

latex

```latex
\begin{verbatim}
  原样输出的文本
\end{verbatim}

\verb|行内原样文本|
```

## 10. 常用宏包

latex

```latex
\usepackage{amsmath}  % 数学增强
\usepackage{amssymb}  % 数学符号
\usepackage{graphicx}  % 图片插入
\usepackage{color}  % 颜色支持
\usepackage{xcolor}  % 扩展颜色
\usepackage{hyperref}  % 超链接
\usepackage{geometry}  % 页面设置
\usepackage{fancyhdr}  % 页眉页脚
\usepackage{listings}  % 代码高亮
\usepackage{algorithm}  % 算法
\usepackage{booktabs}  % 专业表格
\usepackage{multirow}  % 表格合并
\usepackage{subfigure}  % 子图
```