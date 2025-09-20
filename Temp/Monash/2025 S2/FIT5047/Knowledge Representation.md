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
    
    - 用大写字母表示：A,B,C,…A, B, C, \ldotsA,B,C,…
        
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
比如：$$P \rightarrow Q$$

| P   | Q   | $P\rightarrow Q$ |
| --- | --- | ---------------- |
| T   | T   | T                |
| T   | F   | F                |
| F   | T   | T                |
| F   | F   | T                |
$$(P∧Q)→R$$

| P   | Q   | R   | $P \land Q$ | $(P \land Q) \rightarrow R$ |
| --- | --- | --- | ----------- | --------------------------- |
| T   | T   | T   | T           | T                           |
| T   | T   | F   | T           | F                           |
| T   | F   | T   | F           | T                           |
| F   | T   | T   | F           | T                           |
| F   | F   | F   | F           | T                           |
![[Pasted image 20250826145431.png]]
### 模型类别
- **Satisfiable（可满足）**：if it is true in some model
- **Unsatisfiable（不可满足）**：if it is true in no model
- **Valid（有效/永真式, tautology）**：A sentence is valid if it is true in all models

### 逻辑计算公式
![[Pasted image 20250826145536.png]]
![[Pasted image 20250826145732.png]]

# Horn Clauses
**定义:**
- 在一个子句中，**最多只能有一个正文字**（正文字：没有被否定的原子命题）
- **全是负文字的子句**，例如：¬A ∨ ¬B
- **仅有一个正文字，其余为负文字的子句**，例如：¬A ∨ ¬B ∨ C

# Inference Rules
### 1. **Modus Ponens（肯定前件）**

**形式：**

- 如果 P→Q 并且 P 为真，则 Q 为真。

**例子：**

- 前提1：如果下雨（P），地面就湿（Q）。
- 前提2：下雨了（P）。
- 推理结果：地面湿了（Q）。

---

### 2. **Modus Tollens（否定后件）**

**形式：**

- 如果 P→Q 并且 ¬Q 为真，则 ¬P 为真。

**例子：**

- 前提1：如果有火（P），就有烟（Q）。
- 前提2：没有烟（(\neg Q)）。
- 推理结果：没有火（(\neg P)）。

---

### 3. **Disjunctive Syllogism（析取三段论）**

**形式：**

- P∨Q, ¬P ⟹ Q

**例子：**

- 前提1：小明会唱歌或跳舞（P或Q）。
- 前提2：小明不会唱歌（(\neg P)）。
- 推理结果：小明会跳舞（Q）。

---

### 4. **Hypothetical Syllogism（假言三段论）**

**形式：**

- P→Q, Q→R ⟹ P→R

**例子：**

- 前提1：如果学习（P），就会及格（Q）。
- 前提2：如果及格（Q），父母就高兴（R）。
- 推理结果：如果学习（P），父母就高兴（R）。

---

### 5. **Resolution（归结规则）**（常用于自动推理和AI）

**形式：**

- P∨Q, ¬Q∨R ⟹ P∨R

**例子：**

- 前提1：小红去游泳或者下雨（P或Q）。
- 前提2：不是下雨或者要戴伞（(\neg Q)或R）。
- 推理结果：小红去游泳或者要戴伞（P或R）。

