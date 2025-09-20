---
date: 2025-09-20
author:
  - Siyuan Liu
tags:
  - FIT5047
---
# Knowledge Representation
**理解:**
如何用形式化的方法，把现实世界中的知识存储在计算机系统中，使机器能够“理解”、推理和使用这些知识来解决问题

**表达方式:**
- **逻辑表示（Logic Representation）**
    - **命题逻辑**（Propositional Logic）：只表示真/假事实。
    - **谓词逻辑**（Predicate Logic）：可以表示对象及其关系，更强表达力。
- **Model**
	- 模型是一种关于评估**真/假**的形式化表示
	- If a sentence α is true in a model m, we say that m is a model of α, or m satisfies α
---

## Logical Entailment（逻辑蕴涵）

**类别:**
- **Syntactic entailment（$\vdash$）**：通过推理规则（如自然演绎、分辨率等）能够推出
- **Semantic entailment（$\models$）**：基于所有可能模型的真值分布来判断
## Propositional Logic（命题逻辑）
**定义:**
即可以明确判断为 **真 (True)** 或 **假 (False)** 的语句

**基本组成:**
-  **命题变量（Propositional variables）**
    
    - 用大写字母表示：A,B,C,…
        
    - 每个变量的取值为 **真 (T)** 或 **假 (F)**。
        
-  **逻辑连接词（Logical connectives）**
    - 否定（Negation）：$\neg P$
	    - ¬P 为真，当且仅当 P 为假
    - 合取（Conjunction）：$P \land Q$
	    - P∧Q 为真，当且仅当 P 和 Q 都为真
    - 析取（Disjunction）：$P \lor Q$
	    - P∨Q 为真，当且仅当 P 或 Q 至少一个为真
    - 蕴涵（Implication）：$P \rightarrow Q$
	    - P→Q 为假 **当且仅当** P 为真而 Q 为假，其余情况都为真
    - 双条件（Biconditional）：$P \leftrightarrow Q$
	    - P↔Q 为真，当且仅当 P 与 Q 取值相同
-  **复合命题（Compound propositions）**
    - 由命题变量通过逻辑连接词组合而成。
    - 例如：$(P \land Q) \rightarrow \neg R$

### 真值表（Truth Table）

命题逻辑的语义通常通过真值表来定义。  
比如：![[Pasted image 20250826145431.png]]
### 模型类别
- **Satisfiable（可满足）**：if it is true in some model
- **Unsatisfiable（不可满足）**：if it is true in no model
- **Valid（有效/永真式, tautology）**：A sentence is valid if it is true in all models

### 逻辑计算公式
![[Pasted image 20250826145536.png]]
**简单来说：**
- 只有∧，∨适用**交换律，分配律，结合律，迪摩根定律**
- 特殊：蕴涵的对位性
- 特殊：蕴含的消失性
- 特殊：双条件的消失性

### Inference Rule
![[Pasted image 20250826145732.png]]
**从前提得到结论，前提为真则结论为真，前提为假则结论为假**

---
## Conjunctive Normal Form - CNF(合取范式)
**定义：**
命题公式被写成若干子句的合取（AND），每个子句是若干文字的析取（OR）

**举例：**
转化$A↔(B∨C)$为CNF：
1. $(A→(B∨C)) ∧ ((B∨C)→A)$
2. $(¬ A ∨ (B ∨ C)) ∧ (¬ (B ∨ C) ∨ A)$
3. $(¬ A ∨ B ∨ C) ∧ (¬ B ∧ ¬ C ∨ A)$
4. $(¬A ∨ B ∨ C) ∧ (¬ B ∨ A) ∧ (¬ C ∨ A)$
## Horn Clauses
**定义:**
- 在一个子句中，**最多只能有一个正文字**（正文字：没有被否定的原子命题）
- **全是负文字的子句**，例如：¬A ∨ ¬B
- **仅有一个正文字，其余为负文字的子句**，例如：¬A ∨ ¬B ∨ C

## Backward Chaining & Forward Chaining
**举例：**
- R1: A ∧ B → C
- R2: C → D 
- 事实：A, B 
- 目标：证明 D
### Forward：
已知 {A,B} 触发 R1 得 C；已知 {A,B,C} 触发 R2 得 D；证得D

优点：
- 得到所有可推事实；对多目标重用中间结果
- sound and complete for Horn KBs
缺点：
- 若目标很少，可能做了大量与目标无关的推理
### Backward：
要证得D，用 R2 需要 C；要证 C，用 R1 需 A、B；查事实库有 A、B，因此 C、D 成立

优点：
- 聚焦目标，搜索复杂度更小
- sound and complete for Horn KBs
缺点：
- 易陷入递归/环路；没有记忆化时重复子目标代价大