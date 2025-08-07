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
==这种Agent基于当前的感知（当前状态）直接做出反应。它有一个“如果...那么...”的规则集合==
- **特点:**
    - 简单、直接。
    - 不维护任何内部状态或历史信息。
    - 只能对当前感知做出反应，无法处理部分可观测的环境。
### Model based
- 这种Agent维护一个内部模型，描述环境如何运作。它会根据感知更新模型，并基于模型选择行动。
- **特点:**
    - 比简单反射Agent更复杂。
    - 能够处理部分可观测的环境，因为它可以通过模型推断出环境的隐藏状态。
    - 需要维护和更新模型，这可能需要计算资源。
### Goal based
- 这种Agent有一个明确的目标，它会选择能够最快达到目标的行动。
- **特点:**
    - 需要知道目标是什么。
    - 需要搜索和规划能力，找到达到目标的最佳路径。
    - 比基于模型的Agent更智能，因为它知道自己想要什么。
### Utility based
- 这种Agent不仅有目标，还有效用函数，用于评估不同状态的“好坏”。它会选择能够最大化期望效用的行动。
- **特点:**
    - 比基于目标的Agent更灵活，因为它可以在多个目标之间进行权衡。
    - 效用函数可以考虑各种因素，例如成本、风险、时间等。
    - 需要学习或估计效用函数。
### Learning (performance elem +critic +learning element +problem generator)
- observes the world & informs learning elem formulates new driving rules based on the feedback from critic + perf agent knowledge might suggest some driving exercises
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