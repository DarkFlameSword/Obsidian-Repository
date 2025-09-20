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
