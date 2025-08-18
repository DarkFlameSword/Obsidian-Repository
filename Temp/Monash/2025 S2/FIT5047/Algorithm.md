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
1. 将起始节点 u 放入队列中，并标记为已访问
2. 从队列中取出一个节点，访问它并将其所有的未访问邻接节点 v 放入队列中
3. 标记已访问的节点 v，以避免重复访问
4. 重复步骤 2∼3，直到队列为空或找到目标节点

==适用场景:==
如果您想找到`从所有位置出发`或`到达所有位置`的路径, 并且所有位置的移动成本相同，则使用`BFS`

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
###### `Dijkstra`
==理解:==
广度优先算法的拓展算法，广度优先算法每走一步cost都是1，``Dijkstra`考虑不同的cost，并引入优先队列
==适用场景:==
如果您想找到`从所有位置出发`或`到达所有位置`的路径, 并且移动成本不同，则使用`Dijkstra`
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
#### Greedy Best-first Search 贪婪最佳优先搜索
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
==理解:==
每次扩展“估价函数f(n)”最小的节点，f(n)=g(n)+h(n)。
- g(n)：从起点到当前节点n的实际代价（已知）。
- h(n)：从n到目标的启发式估价（用来预测，通常用启发函数估算）
##### A*
==理解==:
`Dijkstra`与`Greedy Best-first Search`的结合, 同时属于A算法下的特例

==特征:==
- h(n)必须是**admissible**: 即h(n)永远不能高估从n到目标的真实最小代价。
- h(n)必须具有**Monotonicity**: $$∀n|\; h(n) ≤ c(n,m) + h(m) \;\text{where m is any child of n}$$
    - `h(n)`: 从节点 n 到目标的估计成本。
    - `c(n,m)`: 从节点 n 移动到其子节点 m 的实际成本。
    - `h(m)`: 从子节点 m 到目标的估计成本。
```
也就是说从节点 `n` 到目标的估计成本，不应该大于“先走到邻居 `m` 的实际成本 `c(n,m)`，再加上从 `m` 到目标的估计成本 `h(m)`”之和
```
- 在极端情况下，当h(n)始终为0，则将由g(n)决定节点的优先级，此时算法就退化成了`Dijkstra`算法
- 在极端情况下，当g(n)始终为0，则将由h(n)决定节点的优先级，此时算法就退化成了`Greedy Best-first`算法

##### UCS (Uniform-Cost Search)
==理解:==
基于`BFS`的一种扩展, 不同于BFS的FIFO, UCS通过cost的优先队列实现

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
理解: 
选择一条策略，使得在对方也“尽力阻止你”的前提下，你的结果最好


### Minimax
#### 理解: 
- 从当前局面出发，展开所有可能的走法，形成 **Game Tree（博弈树）**。
- 在 **叶子节点（终局状态）** 赋予一个 **效用值 (utility)**：
    - MAX 胜利 → 正数（例如 +1 或 +∞）
    - MIN 胜利 → 负数（例如 −1 或 −∞）
    - 平局 → 0
- 从底部开始往回推：
    - 轮到 **MAX** 时，取 **子节点中的最大值**。
    - 轮到 **MIN** 时，取 **子节点中的最小值**。
- 根节点的值就是这个局面的 **Minimax 值**。
- **MAX 玩家** 在根节点选择通向 **最大 Minimax 值** 的分支。

#### 使用场景: 
两人零和博弈（一个人赢，另一个人必然输）

#### Pseudocode
```
function Minimax-Decision(state) returns an action
    return argmax_a ∈ Actions(state) Min-Value(Result(state, a))


function Max-Value(state) returns a utility value
    if Terminal-Test(state) then
        return Utility(state)
    v ← -∞
    for each a in Actions(state) do
        v ← Max(v, Min-Value(Result(state, a)))
    return v


function Min-Value(state) returns a utility value
    if Terminal-Test(state) then
        return Utility(state)
    v ← +∞
    for each a in Actions(state) do
        v ← Min(v, Max-Value(Result(state, a)))
    return v
```

### α-β pruning
#### 理解:
- Minimax虽然能找最优解，但当“棋盘很大”时（如国际象棋），它会枚举无数分支，速度极慢。
- α-β剪枝是一种“提前剪掉不需要计算的分支”的优化方法。
- 简单说，**如果某一分支已经确定不可能比当前最优更好，就直接跳过、不再深入。**
#### Concept:
1. α-value
    - 出现在 **MAX 节点**
    - 目前为止，MAX 在这个节点能保证的 **最大收益下界**
2. β-value
    - - 出现在 **MIN 节点**
    - 目前为止，MIN 在这个节点能保证的 **最小收益上界**
3. 如果在搜索中发现 **α ≥ β**，说明后续分支再探索也没意义，可以剪掉
    - 因为 MAX 至少能保证 α
    - MIN 至多能保证 β
    - 如果 α ≥ β，就代表父节点不会选择这个分支
#### Pruning Rules
##### 1. α cut-off （Alpha 剪枝）

**定义：**
- 如果某个 **MIN 节点** 的 **β ≤ 其任一祖先 MAX 节点的 α**，
- 那么这个 MIN 节点的剩余分支就 **不需要继续搜索**。

**为什么？**
- MIN 要选择最小值。
- 但现在这个 MIN 节点的 **最好情况（β）** 也已经 ≤ 上面某个 MAX 的 α。
- 对那个祖先 MAX 来说，已有的选择已经比这个 MIN 节点更好，所以 MAX 不会考虑这条路径。
- 所以，继续搜索这个 MIN 节点的其他子节点是 **浪费时间**。
##### 2. β cut-off （Beta 剪枝）
**定义：**
- 如果某个 **MAX 节点** 的 **α ≥ 其任一祖先 MIN 节点的 β**，
- 那么这个 MAX 节点的剩余分支就 **不需要继续搜索**。

**为什么？**
- MAX 要选择最大值。
- 但现在这个 MAX 节点的 **最好情况（α）** 已经 ≥ 上面某个 MIN 的 β。
- 对那个祖先 MIN 来说，它已经有更小（更差）的选择，所以不会选择这条路径。  
- 所以，继续搜索这个 MAX 节点的其他子节点也是 **浪费时间**。

#### Pseudocode
```
function ALPHA-BETA-SEARCH(state) returns an action
    v ← MAX-VALUE(state, -∞, +∞)
    return the action in ACTIONS(state) with value v

function MAX-VALUE(state, α, β) returns a utility value
    if TERMINAL-TEST(state) then return UTILITY(state)
    v ← -∞
    for each a in ACTIONS(state) do
        v ← MAX(v, MIN-VALUE(RESULT(state, a), α, β))
        if v ≥ β then return v  // β cut-off
        α ← MAX(α, v)
    return v

function MIN-VALUE(state, α, β) returns a utility value
    if TERMINAL-TEST(state) then return UTILITY(state)
    v ← +∞
    for each a in ACTIONS(state) do
        v ← MIN(v, MAX-VALUE(RESULT(state, a), α, β))
        if v ≤ α then return v  // α cut-off
        β ← MIN(β, v)
    return v
```

#### Additional
==α-β 剪枝算法的性能好坏，与它探索游戏状态（或称节点）的顺序息息相关==

1. α-β 剪枝算法的效率取决于节点被检查的顺序。
    - **详细说明**：如果算法能“猜”对并优先探索那些对玩家最有利的“好棋”，它就能更快地找到一个强有力的剪枝边界（α 或 β 值），从而剪掉更多的分支，大幅提升搜索效率。反之，如果总是先看不好的棋，剪枝效果就会大打折扣。
2. 在“完美排序”的理想情况下，时间复杂度为 $O(b^{\frac{m}{2}} )$
    - **完美排序 (Perfect Ordering)**：这是一个理论上的理想情况，指的是在搜索树的每一层，我们总是先探索最好的那个走法。
    - **b (Branching Factor)**：分支因子，即在每个游戏状态下，平均有多少种可能的走法。
    - **m (Maximum Depth)**：搜索的最大深度。
    - $O(b^m)$：这是原始 Minimax 算法（没有剪枝）的复杂度。
    - $O(b^{\frac{m}{2}} )$：这是 α-β 剪枝在最理想情况下的复杂度。你可以把它看作是$\sqrt{b}^m$
3. 在同等计算资源下搜索的深度可以加倍
    - 这是 α-β 剪枝最惊人的优点。时间复杂度从 `b` 的 `m` 次方降低到了 `b` 的 `m/2` 次方。这意味着，如果原先的计算资源只能让你搜索 6 层深 (b^6)，现在利用 α-β 剪枝，你可能可以搜索到 12 层深 (b^(12/2) = b^6)，因为它们消耗的计算量是近似的。在棋类游戏中，能多看几步往往意味着棋力的巨大提升
4. 在实际应用中，通过引入动态排序方案，可以使算法的性能接近这个理论上的最优值
    - 现实中我们无法做到“完美排序”，因为如果能做到，就意味着我们已经知道哪个是最佳走法了，也就不需要搜索了。因此，我们采用一些**启发式 (Heuristics)** 的方法来对节点的顺序进行**动态排序**，例如：
    - 优先探索那些由上一步迭代确定的最佳走法。
    - 优先探索那些能吃掉对方重要棋子的走法。
    - 使用一个快速的评估函数对所有可能的走法进行初步打分，然后按分数高低进行探索。

### ### Heuristic α-β
#### 适用场景
当计算量过大, 无法找到`Terminnal State`
#### ### Heuristic Evaluation Function
==简单来说，启发式函数为算法提供了一个“人造”的终点，让它在无法看到真正结局时，也能基于“专家经验”做出判断==
1. **设定一个搜索深度限制 (Depth Limit)**，比如 "只往下搜索 6 步"。
2. 算法从当前状态开始向下递归搜索。
3. 在递归的每一步，检查是否满足**终止条件**。这里的终止条件**不再只是**“是否为游戏终局”，而是：
    - **是游戏终局吗？(Is it a terminal state?)** 如果是，返回其真实的效用值 (win/loss/draw)。
    - **达到搜索深度限制了吗？(Has the depth limit been reached?)** 如果是，**调用启发式评估函数 `EVAL(state)`**，并返回评估得到的分数。
4. 这个由启发式函数返回的分数，在 Alpha-Beta 算法的后续步骤中，被当作是终局的效用值来使用，用于更新 `α` 和 `β` 的值，并进行剪枝

### Monte Carlo Simulation(蒙特卡洛模拟)

# Basic Algorithm
## Recursion
==递归步骤==
1. **写出递推公式**：找到将原问题分解为子问题的规律，并且根据规律写出递推公式。
2. **明确终止条件**：推敲出递归的终止条件，以及递归终止时的处理方法。
3. **将递推公式和终止条件翻译成代码**：
    1. 定义递归函数（明确函数意义、传入参数、返回结果等）。
    2. 书写递归主体（提取重复的逻辑，缩小问题规模）。
    3. 明确递归终止条件（给出递归终止条件，以及递归终止时的处理方法）。