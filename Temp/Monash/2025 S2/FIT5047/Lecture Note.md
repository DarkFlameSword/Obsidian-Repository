---
date: 2025-07-29
author:
  - Siyuan Liu
tags:
  - FIT5047
aliases:
  - note
---
# Agent
==An agent is anything that can be viewed as perceiving its environment using sensors, and acting upon that environment via actuators==
## Agent Type
### Simple reflex
`这种Agent基于当前的感知（当前状态）直接做出反应。它有一个“如果...那么...”的规则集合`
**特点:**
    - 简单、直接。
    - 不维护任何内部状态或历史信息。
    - 只能对当前感知做出反应，无法处理部分可观测的环境。
### Model based
`这种Agent维护一个内部模型，描述环境如何运作。它会根据感知更新模型，并基于模型选择行动`
**特点:**
    - 比简单反射Agent更复杂。
    - 能够处理部分可观测的环境，因为它可以通过模型推断出环境的隐藏状态。
    - 需要维护和更新模型，这可能需要计算资源。
### Goal based
`这种Agent有一个明确的目标，它会选择能够最快达到目标的行动`
**特点:**
    - 需要知道目标是什么。
    - 需要搜索和规划能力，找到达到目标的最佳路径。
    - 比基于模型的Agent更智能，因为它知道自己想要什么。
### Utility based
`这种Agent不仅有目标，还有效用函数，用于评估不同状态的“好坏”。它会选择能够最大化期望效用的行动`
**特点:**
    - 比基于目标的Agent更灵活，因为它可以在多个目标之间进行权衡。
    - 效用函数可以考虑各种因素，例如成本、风险、时间等。
    - 需要学习或估计效用函数。
### Learning (performance element +critic +learning element +problem generator)
`学习Agent是一种能够通过经验改进自身性能的Agent。 它与之前提到的Agent类型不同，因为它不仅仅是基于预先设定的规则或模型来行动，而是能够从环境中学习并调整自己的行为`

学习Agent通常由以下几个关键组件组成：

- **Performance Element (性能元件):** 这是Agent的核心部分，负责选择外部行动。它接收感知输入，并根据已有的知识和策略做出决策。可以将性能元件看作是前面提到的几种Agent类型（Simple Reflex, Model-Based, Goal-Based, Utility-Based）之一，它负责实际的行动。
- **Critic (评价器):** 评价器接收性能元件的历史记录，并给出一个反馈信号，说明Agent的表现如何。这个反馈信号可以是一个奖励或惩罚，用于衡量Agent的行动是否成功。简单来说，评价器告诉Agent“你做得好”或“你做得不好”。
- **Learning Element (学习元件):** 学习元件负责根据评价器的反馈信号改进性能元件。它可以调整性能元件的规则、模型、目标或效用函数，使其在未来能够做出更好的决策。学习元件是学习Agent的核心，它利用经验来提升Agent的性能。
- **Problem Generator (问题生成器):** 问题生成器负责提出新的、具有挑战性的问题或探索性的行动，以便Agent能够学习到新的知识。它可以主动探索环境，尝试不同的行动，并观察结果。问题生成器的作用是帮助Agent打破已有的模式，发现新的可能性。
# Rationality
==Rationality depends on (PEAS)==
## PEAS
### Performance Measure
系统需要达到的目标，例如准确率、速度、成本等
### Environment
系统所处的环境，包括输入数据的类型、噪声水平、变化频率等
#### Environment Type
1. Fully / partially observable: An agent's sensors give it access to the complete state of the environment at all times
2. Known / unknown: An agent knows the “laws” of the environment
3. Single / multi agent: An agent operating by itself in an environment
4. Deterministic / stochastic: The next state is completely determined by the current state and the action executed by the agent
5. Episodic / sequential: The agent's experience is divided into atomic episodes. The next episode does NOT depend on previous actions
6. Static / dynamic: The environment is unchanged while an agent is deliberating
7. Discrete / continuous: Pertains to number of states, the way time is handled, and number of percepts and actions
### Actuators
系统可以采取的行动，例如控制机器人、推荐商品、预测结果等
### Sensors
系统用来感知环境的手段，例如摄像头、麦克风、传感器、数据库等

==举例说明==
**1. 自动驾驶汽车：**
- **P**erformance：安全驾驶，准时到达目的地，减少交通事故。
- **E**nvironment：城市道路、高速公路、天气状况（晴天、雨天、雪天）、交通状况（拥堵、畅通）。
- **A**ctuators：方向盘、油门、刹车、转向灯。
- **S**ensors：摄像头、激光雷达、雷达、GPS、惯性测量单元（IMU）。

# Types of Games
1. Perfect information (deterministic, full observe.)
    Chess, Go, noughts-and-crosses (tic-tac-toe), draught (checkers), etc.
2. Imperfect information (stochastic, partially observe.)
    Poker, Texas hold’em (variant of poker), bridge, backgammon, etc
3. n-Players (e.g., n = 2 for chess)
4. Sequential (vs. simultaneous)
5. Zero-sum (vs non-zero sum games)

# Game Trees(博弈树)
## Concept
1. **MAX**：希望最大化自己的得分
2. **MIN**：希望最小化对手的得分
3. A position favorable to MAX → utility > 0
    - 如果某个局面对 MAX 有利，就给它一个 **正数值**，越大越好。
    - MAX 获胜的终局，通常赋值 **+∞**
4. A position favorable to MIN → utility < 0
    - 如果某个局面对 MIN 有利，就给它一个 **负数值**，越小越好
    - MIN 获胜的终局，通常赋值 **−∞**
5. The set of nodes generated for one player is called a **ply**
    - **ply** 表示一层搜索深度：
    - 1 ply = 一个玩家下的一步棋   
    - 2 ply = 双方各下一步（即一个来回）
    - 3 ply = MAX → MIN → MAX
6. Each node has an associated **utility**
    - 每个状态（节点）都有一个 **效用值 (utility)**，用来评估该局面对双方的好坏。
7. **Terminal nodes (终局节点)**
8. 当游戏结束（胜负已定 / 平局），树的叶子节点就是 **terminal nodes**。
    - 它们的 utility 通常是：
    - MAX 胜 → +∞ 或 +1
    - MIN 胜 → −∞ 或 −1
    - 平局 → 0
9. `So`: the initial state of the game
10. `Actions(s)`: moves applicable in state `s`
11. `Result(s,a)`: transition (apply action a in state s)
12. `Is-Terminal(s)`: tests whether the game is over
13. `Utility(s)`: payoff in state s for the current player

# Knowledge Representation
**理解:**
如何用形式化的方法，把现实世界中的知识存储在计算机系统中，使机器能够“理解”、推理和使用这些知识来解决问题

**表达方式:**
- **逻辑表示（Logic Representation）**
    - **命题逻辑**（Propositional Logic）：只表示真/假事实。
    - **谓词逻辑**（Predicate Logic）：可以表示对象及其关系，更强表达力。
- **语义网络（Semantic Networks）**
    - 用**图结构**表示概念和关系，例如：
        - “猫 → 是一种 → 动物”
        - “猫 → 会 → 捉老鼠”
- **框架（Frames）**
    - 类似面向对象的思想，知识表示为属性-值对（slots）。
    - 例如：
		```
		狗：
		类型: 动物
		特征: 四条腿, 会叫
		```
- **产生式规则（Production Rules）**

- 形式为 **IF 条件 THEN 动作/结论**。
	
- 例如：
	
	`IF 天气 = 下雨 THEN 带雨伞`
        
- **本体（Ontology）**
    
    - 一种更系统的表示方法，用于定义领域中的概念、关系和约束。
        
    - 常见于语义网（Semantic Web）、知识图谱（Knowledge Graph）。
        
- **神经网络嵌入（Neural Representation）**
    
    - 在深度学习中，用向量或张量表示知识，例如 word embeddings、graph embeddings。
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
