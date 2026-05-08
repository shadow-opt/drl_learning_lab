# 02 强化学习数学基础课

## 这章为什么存在

00 章讲了训练循环，01 章讲了 PyTorch tensor。02 章开始进入强化学习：智能体不再拿固定标签做监督学习，而是在环境里做 action，收到 reward，并关心长期 return。

本章不假设你会强化学习数学。你只需要先理解一个小网格世界：站在哪个格子是 state，往哪个方向走是 action，走一步收到 reward，直到终点 terminal。value iteration 会用 Bellman backup 计算每个 state 的长期价值，并得到 policy。

## state：智能体当前看到或处在的情况

白话解释：state 是环境当前情况的编号或描述。在本章 GridWorld 中，4x4 网格一共有 16 个 state，用 0 到 15 表示。

最小例子：

```text
0   1   2   3
4   5   6   7
8   9   10  11
12  13  14  15
```

`state=0` 是左上角，`state=15` 是右下角终点。

常见误解：state 不一定是图像。它可以是整数、向量、图像、机器人传感器读数。当前章节用整数，是为了先学清楚数学。

和 DRL 的关系：DQN 会把 state 或 observation 输入神经网络，输出每个 action 的 Q value。

## action：智能体能做的选择

白话解释：action 是智能体在当前 state 可以执行的动作。本章有 4 个离散动作。

最小例子：

```text
0=up, 1=right, 2=down, 3=left
```

常见误解：policy 矩阵里的数字不是 value，也不是 reward，而是 action id。看到 `1` 要翻译成 `right`。

和 DRL 的关系：离散动作算法会输出 action id 或每个 action 的分数；连续控制算法会输出连续 action 向量。

## reward：一步反馈

白话解释：reward 是做完一个 action 后立刻收到的反馈。本章默认每走一步 reward 是 `-1`，走到终点 reward 是 `0`。

最小例子：从 state 14 往右走到终点 15：

```text
next_state=15, reward=0.0, done=True
```

从 state 10 往右走到 11：

```text
next_state=11, reward=-1.0, done=False
```

常见误解：reward 不是总分。它只是一小步的反馈。

和 DRL 的关系：环境每一步给 reward；算法把很多步的 reward 组合成 return 或 target。

## return：从现在开始的长期累计回报

白话解释：return 是从当前时刻往后，把未来 reward 累积起来。带折扣时，越远的 reward 权重越小。

最小例子：如果接下来三步 reward 是 `-1, -1, 0`，且 `gamma=1.0`：

```text
return = -1 + -1 + 0 = -2
```

如果 `gamma=0.9`：

```text
return = -1 + 0.9 * (-1) + 0.9^2 * 0 = -1.9
```

常见误解：reward 和 return 不是同一个东西。reward 是一步，return 是长期。

和 DRL 的关系：policy gradient 常用 return 或 advantage 来评价动作好坏；value network 学的通常也是某种期望 return。

## value：一个 state 长期有多好

白话解释：value 是“从这个 state 开始，按某种策略或最优方式走，未来 return 大概是多少”。在本章每步扣 1 分，所以离终点越近，value 通常越高，也就是不那么负。

最小例子：state 14 在终点左边，向右一步结束，return 是 `0`，所以它的最优 value 是 `0`。state 13 最少还要两步到终点，return 是 `-1 + 0 = -1`。

常见误解：value 高不一定是正数。在每步惩罚的任务里，`-1` 比 `-5` 高，因为损失更少。

和 DRL 的关系：critic、V 网络、Q 网络都在估计长期价值，只是输入输出形式不同。

## policy：每个 state 该做什么

白话解释：policy 是行动规则。离散表格环境里，可以把 policy 写成“每个 state 对应一个 action id”。

最小例子：

```text
policy[14] = 1
```

结合动作图例，`1=right`，意思是在 state 14 往右走。

常见误解：policy 矩阵里的 `2` 不是“价值为 2”，而是 `2=down`。

和 DRL 的关系：PPO、VPG、SAC 的 actor 本身就是 policy；DQN 虽然学 Q value，但执行时也会从 Q value 里派生 greedy policy。

## Bellman backup：用下一步更新当前价值

白话解释：Bellman backup 是强化学习最核心的递推思想：当前 state 的好坏，等于“这一步 reward + 下一步 state 的价值”。如果有多个 action，就试所有 action，选最好的。

最小公式：

```text
value[state] = max_action (reward + gamma * value[next_state])
```

如果 `next_state` 是 terminal，就不能继续 bootstrap：

```text
value[state] = reward
```

最小手算：从 state 14 向右到 terminal 15，reward 是 `0`，done 是 True：

```text
backup = 0
```

从 state 13 向右到 state 14，reward 是 `-1`，假设 `value[14] = 0`，`gamma=1`：

```text
backup = -1 + 1 * 0 = -1
```

常见误解：不管 done 都加 `gamma * value[next_state]`。terminal 后 episode 已结束，不能再把终点之后的价值加进来。

和 DRL 的关系：DQN 的 target 也是 Bellman backup：`reward + gamma * max_next_q`，done 时去掉 bootstrap。

## terminal 和 bootstrap

白话解释：terminal 是 episode 结束状态。bootstrap 是用自己当前估计的 `value[next_state]` 来帮助估计当前 state。

最小例子：

```python
bootstrap = 0.0 if done else gamma * values[next_state]
action_value = reward + bootstrap
```

常见误解：bootstrap 是“作弊使用未来真实答案”。实际上它使用的是当前 value 表或 value network 的估计。

和 DRL 的关系：几乎所有 TD 类算法都要处理 done mask。done mask 错了，会让算法以为终点之后还能继续拿价值。

## value iteration 在做什么

白话解释：value iteration 反复扫过所有 state。每个 state 试四个 action，用 Bellman backup 算 action value，选最大的更新当前 value，同时记录最好的 action 到 policy。

流程：

```text
初始化 values 全 0
重复很多轮：
  对每个非 terminal state：
    对每个 action：
      transition 得到 next_state, reward, done
      action_value = reward + bootstrap
    values[state] = 最大 action_value
    policy[state] = 最大 action_value 对应的 action
  如果本轮变化很小，就停止
```

常见误解：value iteration 不是用神经网络训练，也不是采样很多 episode。它是已知环境转移时的动态规划算法。

和 DRL 的关系：表格 value iteration 是理解 DQN target、TD learning 和 policy improvement 的前置直觉。

## 学完检查

你应该能回答：

1. reward 和 return 的区别。
2. value 为什么可以是负数但仍然有大小比较。
3. policy 矩阵里的数字怎么翻译成动作。
4. Bellman backup 怎么手算。
5. terminal 时为什么不能继续 bootstrap。

## 公开课扩展：形式化定义

一个有限 MDP 可以写成五元组：

```text
M = (S, A, P, R, gamma)
```

- `S`：state 集合。
- `A`：action 集合。
- `P(s' | s, a)`：在 state `s` 做 action `a` 后转移到 `s'` 的概率。
- `R(s, a, s')`：这一步转移得到的 reward。
- `gamma`：discount factor，范围 `[0, 1]`。

本章 GridWorld 是确定性 MDP：给定 `(s, a)` 后，`s'` 是确定的，所以 `P` 的某一个 next state 概率为 1，其余为 0。

## 公开课扩展：trajectory 与 return

trajectory 是一段交互序列：

```text
s_0, a_0, r_1, s_1, a_1, r_2, ..., s_T
```

从时间 `t` 开始的 discounted return：

```text
G_t = r_{t+1} + gamma r_{t+2} + gamma^2 r_{t+3} + ...
```

注意下标：`r_{t+1}` 是从 `s_t` 做 `a_t` 后收到的 reward。很多初学错误来自把 state 的编号和 reward 的时间下标混在一起。

## 公开课扩展：value、Q 与 policy

给定 policy `pi`，状态价值函数是：

```text
V^pi(s) = E_pi[G_t | s_t = s]
```

动作价值函数是：

```text
Q^pi(s, a) = E_pi[G_t | s_t = s, a_t = a]
```

最优价值函数：

```text
V*(s) = max_pi V^pi(s)
Q*(s, a) = max_pi Q^pi(s, a)
```

本章 `value_iteration` 返回的是 `V*` 的表格近似和对应 greedy policy。后续 DQN 直接学习 `Q*(s, a)` 的神经网络近似。

## 公开课扩展：Bellman expectation equation

对固定 policy，Bellman expectation equation 是：

```text
V^pi(s) = sum_a pi(a|s) sum_s' P(s'|s,a) [R(s,a,s') + gamma V^pi(s')]
```

白话：一个 state 的价值等于按 policy 选动作、按环境转移到下一状态后，当前 reward 加未来 value 的期望。

确定性 policy 和确定性环境会让求和消失，只剩：

```text
V^pi(s) = R(s, pi(s), s') + gamma V^pi(s')
```

## 公开课扩展：Bellman optimality equation

最优控制不固定 policy，而是在每个 state 选最好的 action：

```text
V*(s) = max_a sum_s' P(s'|s,a) [R(s,a,s') + gamma V*(s')]
```

本章代码中的 `action_values[action] = reward + bootstrap` 就是确定性版本的括号项；`action_values.max()` 就是 `max_a`。

## 公开课扩展：terminal mask

terminal state 后 episode 结束，所以 Bellman backup 要变成：

```text
target = reward                         if done
target = reward + gamma * V(next_state) if not done
```

代码写成：

```python
bootstrap = 0.0 if done else gamma * values[next_state]
action_values[action] = reward + bootstrap
```

这个 mask 在 DQN 中同样关键。若 done mask 错了，模型会把终点后的不存在价值加进 target，造成系统性偏差。

## 公开课扩展：Bellman backup 推导

从 return 定义开始：

```text
G_t = r_{t+1} + gamma r_{t+2} + gamma^2 r_{t+3} + ...
```

把第一步 reward 拆出来：

```text
G_t = r_{t+1} + gamma (r_{t+2} + gamma r_{t+3} + ...)
```

括号里的部分就是从 `s_{t+1}` 开始的 return，也就是下一状态 value 的估计对象：

```text
V(s_t) = E[r_{t+1} + gamma V(s_{t+1})]
```

如果要控制而不是评估固定 policy，就在所有 action 里选最大的 backup：

```text
V*(s_t) = max_a E[r_{t+1} + gamma V*(s_{t+1}) | s_t, a]
```

本章 GridWorld 是确定性的，所以期望符号可以落成代码中的单次 `transition`。

## 公开课扩展：例题

例题 1：state 14 向右。

```text
s=14, a=right
s'=15, reward=0, done=True
target = reward = 0
```

因为 `done=True`，bootstrap 被 mask 掉。

例题 2：state 13 向右，已知 `V(14)=0`，`gamma=1`。

```text
s=13, a=right
s'=14, reward=-1, done=False
target = -1 + 1 * V(14) = -1
```

这两个例题解释了为什么 terminal 邻居的 value 可以是 0，而更远一格是 -1。

## 公开课扩展：value iteration 的收敛直觉

Bellman optimality operator 可以看作一个把旧 value 表变成新 value 表的函数：

```text
V_{k+1} = T* V_k
```

当 `gamma < 1` 时，这个 operator 在 sup norm 下是 contraction，因此反复应用会收敛到唯一固定点 `V*`。本章 demo 用 `gamma=1.0`，因为有限 GridWorld 有 terminal 且每步惩罚，仍能在这个简单环境中稳定得到最优值；后续更一般的无限期任务通常要求 `gamma < 1` 来保证理论性质。

## 公开课扩展：本章定理化小结

三个 RL math invariants：

1. Reward/return invariant：一步 reward 不是长期 return。
2. Terminal invariant：done 时 bootstrap 必须为 0。
3. Greedy improvement invariant：给定 value 表，policy 可由最大 action value 提取。

后续算法只是把这些表格量换成采样估计或神经网络估计。数学骨架没有消失。
