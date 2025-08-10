---
date: 2025-08-05
tags:
  - FIT5047
author:
  - Siyuan Liu
aliases:
  - base
---
# Search Strategies

## Tentative Strategy 试探性策略
### Uninformed 盲目搜索
```
算法只知道问题的规则和初始状态，不知道任何目标方向的信息。决策完全基于问题定义，不使用任何额外的启发或提示
```
#### Backtracking 回溯
理解: 追踪一条路径的数据. 专注于**next step** ,如果**next step** 失败则回溯到上一个状态, 然后选择另一条路
```
# 通用模板

res = []    # 存放所欲符合条件结果的集合
path = []   # 存放当前符合条件的结果
def backtracking(nums):             # nums 为选择元素列表
    if 遇到边界条件:                  # 说明找到了一组符合条件的结果
        res.append(path[:])         # 将当前符合条件的结果放入集合中
        return

    for i in range(len(nums)):      # 枚举可选元素列表
        path.append(nums[i])        # 选择元素
        backtracking(nums)          # 递归搜索
        path.pop()                  # 撤销当前选择, 准备寻找下一个符合条件的选择

backtracking(nums)
```

==Attention:==
1. backtrack 只在以下任意一个情况下发生:
- 到达了一个死胡同状态
- 已经没有可以应用的规则
- 生成了一个以前遇到过的状态
- 已经应用了任意数量的规则但仍未达到目标
#### Tree- and Graph search
理解: 追踪多条路线的数据. 从起点出发，沿着所有可能的通路走（比如广度优先、深度优先），直到找到出口。你不提前判断哪条路可能更好，只是机械地遍历所有路径
![[Pasted image 20250804131655.png]]
##### Breadth-first search (BFS)
==特点:==
1.  先进先出
2. 使用队列实现

==逻辑:==
1. 将起始节点 u 放入队列中，并标记为已访问。
2. 从队列中取出一个节点，访问它并将其所有的未访问邻接节点 v 放入队列中。
3. 标记已访问的节点 v，以避免重复访问。
4. 重复步骤 2∼3，直到队列为空或找到目标节点。

```Python
import collections

class Solution:
    def bfs(self, graph, u):
        visited = set()                     # 使用 visited 标记访问过的节点
        queue = collections.deque([])       # 使用 queue 存放临时节点
        
        visited.add(u)                      # 将起始节点 u 标记为已访问
        queue.append(u)                     # 将起始节点 u 加入队列中
        
        while queue:                        # 队列不为空
            u = queue.popleft()             # 取出队头节点 u
            print(u)                        # 访问节点 u
            for v in graph[u]:              # 遍历节点 u 的所有未访问邻接节点 v
                if v not in visited:        # 节点 v 未被访问
                    visited.add(v)          # 将节点 v 标记为已访问
                    queue.append(v)         # 将节点 v 加入队列中
                

graph = {
    "0": ["1", "2"],
    "1": ["0", "2", "3"],
    "2": ["0", "1", "3", "4"],
    "3": ["1", "2", "4", "5"],
    "4": ["2", "3"],
    "5": ["3", "6"],
    "6": []
}

# 基于队列实现的广度优先搜索
Solution().bfs(graph, "0")
```
##### Depth-first search (DFS)
==特点:==
1. 后进先出
2. 使用堆栈/递归实现
==逻辑:==
3. 选择起始节点 u，并将其标记为已访问。
4. 检查当前节点是否为目标节点（看具体题目要求）。
5. 如果当前节点 u 是目标节点，则直接返回结果。
6. 如果当前节点 u 不是目标节点，则遍历当前节点 u 的所有未访问邻接节点。
7. 对每个未访问的邻接节点 v，从节点 v 出发继续进行深度优先搜索（递归）。
8. 如果节点 u 没有未访问的相邻节点，回溯到上一个节点，继续搜索其他路径。
9. 重复 2∼6 步骤，直到遍历完整个图或找到目标节点为止。

```Python
class Solution:
    def dfs_recursive(self, graph, u, visited):
        print(u)                        # 访问节点
        visited.add(u)                  # 节点 u 标记其已访问

        for v in graph[u]:
            if v not in visited:        # 节点 v 未访问过
                # 深度优先搜索遍历节点
                self.dfs_recursive(graph, v, visited)
        

graph = {
    "A": ["B", "C"],
    "B": ["A", "C", "D"],
    "C": ["A", "B", "D", "E"],
    "D": ["B", "C", "E", "F"],
    "E": ["C", "D"],
    "F": ["D", "G"],
    "G": []
}

# 基于递归实现的深度优先搜索
visited = set()
Solution().dfs_recursive(graph, "A", visited)
```

##### Depth-limited search (DLS)
##### Uniform-cost search (UCS)
##### Iterative deepening search (IDS)
### informed [[Heuristic Graph-Search|启发式搜索]]
```
算法利用问题特定的信息或启发函数（heuristic），指导搜索更快地找到解。决策时会估计哪些路径“看起来”更有希望
```
#### Greedy best-first search 贪婪最佳优先搜索
==理解==: 贪心算法是一种改进的「分步解决算法」，其核心思想是：将求解过程分成「若干个步骤」，然后根据题意选择一种「度量标准」，每个步骤都应用「贪心原则」，选取当前状态下「最好 / 最优选择（局部最优解）」，并以此希望最后得出的结果也是「最好 / 最优结果（全局最优解）」

==贪心算法使用条件:==
1. 贪⼼选择性质: 一个问题的全局最优解可以通过一系列局部最优解（贪心选择）来得到
2. 最优子结构: 一个问题的最优解包含其子问题的最优解

==启发函数:==
$$f(n) = h(n)$$
```
贪心算法只关注当前节点到目标节点的最小花销h(n)
```
==步骤:==
1. **初始化**：
    a. 创建一个`Open Set`和一个`Closed Set`
    b. 将起点放入`Open Set`。
2. **循环**：只要`Open Set`不为空，就执行以下操作： 
    a. 从`Open Set`中找到 `h(n)` 最小的节点，称之为 `current`
    b. 将 `current` 从`Open Set`移到`Closed Set`
    c. 检查目标：如果 `current` 就是目标节点，则搜索成功，返回路径。 
    d. 扩展节点：获取 `current` 的所有邻居节点。
    e. 对于每个邻居节点： 
    - 如果它已经在`Open Set`中，则忽略它
    - 如果它不在`Open Set`中，则计算它的 `h(n)` ，设置其父节点为 `current`，然后将它加入`Open Set`
#### A
理解: 每次扩展“估价函数f(n)”最小的节点，f(n)=g(n)+h(n)。
- g(n)：从起点到当前节点n的实际代价（已知）。
- h(n)：从n到目标的启发式估价（用来预测，通常用启发函数估算）
#### A*
理解: A* 算法是A算法的一个特例, A* 对启发函数h(n)有严格要求（必须可采纳）
- h(n)必须是“可采纳的/乐观的/低估的”（admissible）：即h(n)永远不能高估从n到目标的真实最小代价。
- 如果h(n)满足可采纳性，则A*算法保证找到一条**最优路径**。

## Irrevocable Strategy 不可逆策略
### Informed
#### Hill climbing
理解: 不断微调频率，只要信号变强就继续调，直到信号无法再增强为止（但可能只是“局部最优”）
#### Local beam search 局部束搜索
理解: 初始挑选5个人，每轮只保留表现最好的5人，淘汰差的，再各自扩展新候选人，继续选最优
#### Simulated annealing 模拟退火
理解: 爬山法基础上，偶尔允许“走下坡路”，以一定概率接受更差的解，有助于跳出局部最优点
#### Genetic algorithms 遗传算法
理解: 一群解（个体）交叉、变异，优胜劣汰。比如用遗传算法优化神经网络结构，每代都会产生新一批组合，保留表现最好的

## Adversarial search algorithms 对抗性搜索算法

### Optimal decisions
理解: 选择一条策略，使得在对方也“尽力阻止你”的前提下，你的结果最好
### Minimax
理解: 
- 把所有可能的走法用“树”表示出来，自己走一步，对手走一步，依次展开。
- **自己**时，选能让自己得分**最大的分支**（max）。
- **对手**时，假设对方会选能让你得分**最小的分支**（min）。
- 递归回溯，最终选择自己的最优开局。

### α-β pruning
理解:
- Minimax虽然能找最优解，但当“棋盘很大”时（如国际象棋），它会枚举无数分支，速度极慢。
- α-β剪枝是一种“提前剪掉不需要计算的分支”的优化方法。
- 简单说，**如果某一分支已经确定不可能比当前最优更好，就直接跳过、不再深入。**
# Basic Algorithm
## Recursion
==递归步骤==
1. **写出递推公式**：找到将原问题分解为子问题的规律，并且根据规律写出递推公式。
2. **明确终止条件**：推敲出递归的终止条件，以及递归终止时的处理方法。
3. **将递推公式和终止条件翻译成代码**：
    1. 定义递归函数（明确函数意义、传入参数、返回结果等）。
    2. 书写递归主体（提取重复的逻辑，缩小问题规模）。
    3. 明确递归终止条件（给出递归终止条件，以及递归终止时的处理方法）。