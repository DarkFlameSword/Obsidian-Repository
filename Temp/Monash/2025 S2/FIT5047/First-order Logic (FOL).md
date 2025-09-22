---
date: 2025-09-20
author:
  - Siyuan Liu
tags:
  - FIT5047
---
**FOL与Propositional Logic的区别：**
- Propositional Logic只处理“整体命题”的真/假
- FOL能谈论“Object, Relation, Function”，用量词表达“所有/存在”的陈述，更精细、更有表达力

# FOL Syntax
## term
constant, variable, function
like: x、0、succ(x)、plus(x,1)
## formula
### atomic formula
P(t1,…,tn)、t1 = t2
### composite formula
∀x (P(x) → ∃y R(x,y))
## literal
atomic formula
## ground literal
literal without variables

# Well-Formed Formulas - WFF (良构式)
**定义：**
- 良构式是指严格按照形式语法（grammar）由符号构成的、语法合法的公式
- 作用：明确“什么是合法表达”，为语义（真假解释）与推理（证明规则）奠定基础
- 特性：唯一可读性（unique readability）——每个良构式都有唯一的解析树/括号结构

# FOL Equivalences
$$¬( x)P(x) ≡ ( x) [¬P(x)]$$
There does not exist an x such that P(x) is true ≡ For all x, P(x) is false
$$¬( x)P(x) ≡ ( x) [¬P(x)]$$
It is not true that for all x P(x) is true ≡ There exists an x, such that P(x) is false
$$( x)[P(x) Λ Q(x)] ≡ ( x)P(x) Λ ( y)Q(y)$$
For all x, P(x) and Q(x) are true ≡ For all x, P(x) is true, and for all y Q(y) is true
$$( x)[P(x) ν Q(x)] ≡ ( x)P(x) ν ( y)Q(y)$$
There is an x, such that P(x) is true or Q(x) is true ≡ There is an x, such that P(x) is true, or there is a y, such that Q(y) is true

# Substitution
$$ P(x, y) \{x|A, y|B\} → P(A, B)$$
- 花括号里的 {x|A, y|B} 表示一个代换 σ：把 x 替换为 A，把 y 替换为 B
$$\begin{aligned}
& s1 = \{z|g(x,y)\} \\ 
& s2 = \{x|A, y|B, w|C, z|D\} \\
& s1s2 = \{z|g(x,y)\}\{x|A, y|B, w|C, z|D\}=\{z|g(A,B), x|A, y|B, w|C\} \\
& s2s1 = \{x|A, y|B, w|C, z|D\} \{z|g(x,y)\} = \{x|A, y|B, w|C, z|D\}
\end{aligned}$$
对于计算$s_is_j$
1. 寻找$s_i$中所有的变量，对每个变量应用$s_j$的替换，且添加到结果中
2. $s_j$中没有被$s_i$使用的替换需要添加到结果中
# Unification
**定义：**
给定两个（或一组）项/原子式，寻找一个代换 σ，使它们在同时应用 σ 后变得完全相同

**举例：**
E1 = KNOWS(John, x)
E2 = KNOWS(y, Bill)

Unification s = {y/John, x/Bill}

# Converting wffs into Clauses
- literal：原子公式或其否定，如 P(t) 或 ¬P(t)。
- clause：若干文字的析取，如 (¬P(x) ∨ Q(f(x)) ∨ R). 常把子句看成“文字集合”，次序与重复无关。
- 子句集：若干子句的合取（AND），是分辨率算法的输入。

1. 消去 → 与 ↔
    - φ → ψ 等价于 ¬φ ∨ ψ
    - φ ↔ ψ 等价于 (φ→ψ) ∧ (ψ→φ)，再继续化简
2. 否定下推到原子（NNF：否定范式）
    - 用德摩根与量词对偶：¬∀x φ ≡ ∃x ¬φ；¬∃x φ ≡ ∀x ¬φ
    - 直到 ¬ 只作用在原子公式上
3. Skolem 化（消去存在量词，保持“可满足性等价”）
![[Pasted image 20250922162751.png]]
    - 注意：Skolem 化不是语义等价，只是“可满足性等价”，但足够用于反证
5. 删除所有∀
6. Eliminate Λ symbols
    - 用分配律把公式化为“合取的析取”
    - 或者直接提取成括号：P(x) Λ Q(x) = {P(x) , Q(x)}
7. Standardize variables apart
	- $\{P(x) , Q(x)\} → \{P(x_1) , Q(x_2)\}$

---
# General Resolution
子句集：{P∨Q}, {¬Q∨R}, {¬R}

- {¬Q∨R} 与 {¬R} 归结⇒ {¬Q}
- {P∨Q} 与 {¬Q} 归结⇒ {P}
# Resolution-refutation
**Resolution-refutation:**
1. 消去 →, ↔：φ→ψ ≡ ¬φ∨ψ
2. Skolem 化（去 ∃）：用 常量/函数 替代存在变量；删除 ∃
3. 分配 ∨ 到 ∧ 得 CNF
4. 去掉显式 ∀
5. 变量标准化（避免不同量词重名）