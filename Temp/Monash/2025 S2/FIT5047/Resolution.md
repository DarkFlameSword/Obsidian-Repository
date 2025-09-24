---
date: 2025-09-23
author:
  - Siyuan Liu
tags:
  - FIT5047
---
# General Resolution
**概念:**
- General Resolution 是 **命题逻辑/谓词逻辑** 里的一个基本推理规则
- 它允许我们从两个含有 **互补文字（complementary literals）** 的子句推出一个新的子句（称为 _resolvent_）
- 给定两个子句：
    - $\{L_j\}$
    - $\{M_i\}$
- 找到其中某些文字 $\{l_j\}$和 $\{m_i\}$，使得 $\{l_j\}$与 $\{\neg m_i\}$**可以统一**（存在一个最一般合一器，MGU）。
- 那么我们可以推出一个新的子句：
    $\{L_j\} - \{l_j\})\sigma \; \cup \; (\{M_i\} - \{m_i\})\sigma$
    其中 $\sigma$ 是 mgu

**举例:**
1. Everyone who can read is literate:  
    $\forall x [ \neg CANREAD(x) \lor LITERATE(x) ]$
2. Whoever goes to school can read:  
    $\forall x [ \neg GOSCHOOL(x) \lor CANREAD(x) ]$
    
通过归结：
- 子句 1：$\neg CANREAD(x_2) \lor LITERATE(x_2)$
- 子句 2：$\neg GOSCHOOL(x_1) \lor CANREAD(x_1)$

取 mgu \{x_2|x_1\}，得到 resolvent：
$\neg GOSCHOOL(x_1) \lor LITERATE(x_1)$
这就表示：谁上学，谁就有文化（literate）

---
# Resolution Refutation
**概念:**
**Resolution refutation** 是一种 **反证法证明** 技术

**思路:**
1. 要证明某个目标语句 $Q$，先取其否定 $\neg Q$，加到知识库（KB）中
2. 然后不断应用 General Resolution 规则，直到推导出矛盾（空子句 NIL）
3. 推导出矛盾，说明 $\neg Q$不成立，从而原命题 $Q$ 得证
- **Negate the goal**：把要证明的结论取反，加入子句集合。
    
- **Apply resolution**：逐步做归结，推导新子句。
    
- **Contradiction (NIL)**：如果得到了空子句，就说明推理闭合，原命题成立。