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
- brake when brake-lights of car in front light up

### Model based
+ remember the roads travelled, time, state

### Goal based
+ make a plan to reach a destination

### Utility based
+ quickest with least petrol consumption

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