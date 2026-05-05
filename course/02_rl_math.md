# 02 RL Math

## 学习目标

本章建立进入 RL 和 Deep RL 的数学语言。学完后你应该能解释 MDP、state、action、reward、transition、policy、return、value function、Bellman equation、Monte Carlo、Temporal Difference、advantage 的含义，并能手推最基本的 Bellman 更新。

这一章不要急着上神经网络。RL 的核心难点不是网络结构，而是“当前动作影响未来数据分布和长期回报”。Bellman 思想就是把长期问题拆成一步 reward 加未来 value。

## 为什么重要

如果不理解 Bellman equation，DQN 的 TD target 只是一个代码公式；如果不理解 policy/value，VPG 和 PPO 的 loss 看起来就是魔法；如果不理解 bootstrapping，TD、Q-learning、critic 的偏差方差权衡会很难 debug。

RL 数学章的目标是给后面所有算法提供统一语言。DQN、VPG、PPO、DDPG、TD3、SAC 都可以看成对 policy、value、Bellman backup、sampling 和 optimization 的不同组合。

## 核心直觉

Agent 在 state 中选择 action，environment 返回 reward 和 next state。我们关心的不是眼前 reward，而是未来折扣回报：

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
$$

`gamma` 越接近 1，agent 越重视长期回报；越接近 0，越短视。

Value function 是“站在某个位置往后看”的期望回报。Policy 是“在某个位置怎么行动”的规则。RL 算法的基本套路是：估计 value，改进 policy，再用新数据继续估计。

读 RL 公式时不要被符号吓住。`s` 是当前位置，`a` 是当前动作，`s'` 是下一状态，`r` 是一步奖励，`gamma` 是未来折扣。Bellman 公式的核心总是同一句话：当前价值等于一步奖励加上下一个位置的价值。不同算法的差异在于下一个价值怎么估计、是否取最大、是否按 policy 加权、是否等 episode 结束。

## 数学与公式

状态价值函数：

$$
V^\pi(s) = \mathbb{E}_\pi [G_t \mid S_t=s]
$$

动作价值函数：

$$
Q^\pi(s,a) = \mathbb{E}_\pi [G_t \mid S_t=s, A_t=a]
$$

Bellman expectation equation：

$$
V^\pi(s) =
\sum_a \pi(a|s)\sum_{s',r}p(s',r|s,a)
\left[r + \gamma V^\pi(s')\right]
$$

Bellman optimality equation：

$$
V^*(s) =
\max_a \sum_{s',r}p(s',r|s,a)
\left[r + \gamma V^*(s')\right]
$$

TD error：

$$
\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)
$$

Advantage：

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)
$$

Advantage 告诉我们某个 action 比当前 state 的平均水平好多少。PPO、VPG 等 policy gradient 方法会大量使用这个概念。

## 手算样例：一步 Bellman backup

假设在某个 GridWorld state `s` 有两个可选 action：

- `right`：确定到达 `s1`，即时 reward 为 `-1`，当前估计 `V(s1)=4`
- `down`：确定到达 `s2`，即时 reward 为 `-1`，当前估计 `V(s2)=2`
- 折扣 `gamma=0.9`

如果做 Bellman optimality backup：

```text
right: -1 + 0.9 * 4 = 2.6
down:  -1 + 0.9 * 2 = 0.8
V_new(s) = max(2.6, 0.8) = 2.6
policy(s) = right
```

这就是 value iteration 在每个 state 做的事：枚举 action，估计“一步 reward + 下一状态 value”，取最大的那个。DQN 的 TD target 只是把这个动作价值表换成神经网络近似。

如果是固定随机 policy，两个 action 各有 50% 概率，则 Bellman expectation backup 是：

```text
V_new(s) = 0.5 * 2.6 + 0.5 * 0.8 = 1.7
```

所以看到公式里的 `max` 和 `sum_a pi(a|s)`，要立刻问：这是在找最优 policy，还是在评估一个固定 policy？

## 算法流程

Policy evaluation：固定 policy，反复更新 value，直到 value 稳定。

Policy improvement：根据当前 value 选择更好的 action。

Policy iteration：evaluation 和 improvement 交替进行。

Value iteration：直接使用 Bellman optimality backup，每次让 value 朝最优方向更新。

Monte Carlo：等 episode 结束，用真实 return 更新。

Temporal Difference：不等 episode 结束，用一步 reward 加估计 value 更新。

## 代码双索引

- 教学 demo：`labs/02_rl_math/code/value_iteration_demo.py`
- GridWorld：`src/drl_lab/algorithms/tabular/gridworld.py`
- Value iteration：`src/drl_lab/algorithms/tabular/value_iteration.py`
- Policy iteration：`src/drl_lab/algorithms/tabular/policy_iteration.py`
- 测试：`tests/test_tabular.py`

## 实验任务

运行：

```bash
conda run -n drl-lab python labs/02_rl_math/code/value_iteration_demo.py
```

期望看到类似输出：

```text
action legend: 0=up, 1=right, 2=down, 3=left
values:
...
policy:
...
```

观察每个 state 的 value 如何变化。手动选一个非终止 state，写出它的一步 Bellman backup。改变 discount factor，观察 value 是否更短视。

## Debug Checklist

- terminal state 是否仍然被错误 bootstrap。
- reward 是到达下一状态时获得，还是离开当前状态时获得。
- `gamma` 是否小于 1。
- value 更新是同步还是异步，是否和实验解释一致。
- policy evaluation 是否在固定 policy 下进行。
- `V(s)` 和 `Q(s,a)` 是否混用。
- Bellman expectation 是否按 policy 加权，Bellman optimality 是否取最优 action。

## Spinning Up 对照

Spinning Up 的 RL Intro 会使用 policy、value、advantage、Bellman backup 等概念。本章是对那部分的预备和展开。读 Spinning Up 时，重点把其中的 value function 和 advantage 定义映射回本章公式。

## 学完标准

- 能写出 return、V、Q、Bellman expectation、Bellman optimality。
- 能解释 MC 和 TD 的区别。
- 能手动算一个 GridWorld state 的 Bellman backup。
- 能跑通 value iteration demo，并在报告里解释一个 value 变化。

如果卡住：先回到本章“手算样例”，只用两个 action 算一次 backup；再去看 `src/drl_lab/algorithms/tabular/value_iteration.py` 中是不是在做同一件事。
