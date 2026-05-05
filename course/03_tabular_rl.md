# 03 Tabular RL

## 学习目标

本章在没有神经网络的情况下学习 RL update rule。学完后你应该能解释 value iteration、policy iteration、Monte Carlo control、SARSA、Expected SARSA、Q-learning 的区别，并能手写每个方法的核心更新式。

Tabular RL 是 Deep RL 的低维显微镜。DQN 本质上是用神经网络近似 Q-table；actor-critic 本质上也是在学 policy 和 value。先在表格环境中看清 update 逻辑，再进入深度网络会稳很多。

## 为什么重要

神经网络会带来 function approximation、optimizer、batch、target network、replay buffer 等复杂因素。如果你直接从 DQN 开始，TD target 出问题时很难判断是 RL 公式错了，还是 PyTorch 工程错了。Tabular RL 把状态动作空间压成一张表，方便你逐项检查。

本章的重点不是追求复杂环境分数，而是看清三组差异：model-based 和 model-free，on-policy 和 off-policy，MC 和 TD。

## 核心直觉

Value iteration 和 policy iteration 假设你知道环境转移模型，也就是知道每个 action 可能去哪里、奖励是多少。MC、SARSA、Q-learning 不需要完整模型，它们从采样 episode 中学习。

On-policy 方法学习“自己当前行为策略”的价值。SARSA 用实际采样到的 next action 更新，所以是 on-policy。Q-learning 用 next state 中最大的 Q 值更新，即使行为策略还在探索，所以是 off-policy。

## 数学与公式

Monte Carlo control 使用完整 return：

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) +
\alpha \left[G_t - Q(S_t,A_t)\right]
$$

SARSA 更新：

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) +
\alpha \left[R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)\right]
$$

Q-learning 更新：

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) +
\alpha \left[R_{t+1} + \gamma \max_a Q(S_{t+1},a) - Q(S_t,A_t)\right]
$$

Expected SARSA 更新：

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) +
\alpha \left[R_{t+1} + \gamma \sum_a \pi(a|S_{t+1})Q(S_{t+1},a) - Q(S_t,A_t)\right]
$$

这些公式差异只在 target 项，但算法行为会明显不同。SARSA 会把探索风险计入学习；Q-learning 更接近学习 greedy policy 的最优值。

## 算法流程

Tabular control 的通用流程：

1. 初始化 Q-table。
2. 用 epsilon-greedy 选择 action。
3. 执行动作，得到 reward、next state、done。
4. 构造 target。
5. 用 `Q <- Q + alpha * (target - Q)` 更新。
6. episode 结束后记录 return。
7. 降低或保持 epsilon，继续采样。

## 代码双索引

- Value iteration demo：`labs/03_tabular_rl/code/value_iteration_demo.py`
- Policy iteration demo：`labs/03_tabular_rl/code/policy_iteration_demo.py`
- Monte Carlo demo：`labs/03_tabular_rl/code/monte_carlo_control_demo.py`
- TD control demo：`labs/03_tabular_rl/code/td_control_demo.py`
- Q-learning demo：`labs/03_tabular_rl/code/q_learning_demo.py`
- 工程实现：`src/drl_lab/algorithms/tabular`
- 测试：`tests/test_tabular.py`

## 实验任务

运行：

```bash
conda run -n drl-lab python labs/03_tabular_rl/code/value_iteration_demo.py
conda run -n drl-lab python labs/03_tabular_rl/code/policy_iteration_demo.py
conda run -n drl-lab python labs/03_tabular_rl/code/monte_carlo_control_demo.py
conda run -n drl-lab python labs/03_tabular_rl/code/td_control_demo.py
conda run -n drl-lab python labs/03_tabular_rl/code/q_learning_demo.py
```

记录不同算法得到的 policy 是否一致。改变 epsilon 和 alpha，观察收敛速度和稳定性。

## Debug Checklist

- terminal state 是否没有继续 bootstrap。
- epsilon-greedy 是否真的有探索。
- Q-table shape 是否是 `[num_states, num_actions]`。
- SARSA 是否使用实际 next action。
- Q-learning 是否使用 `max_a Q(next_state, a)`。
- Expected SARSA 的 action probability 是否归一化。

## Spinning Up 对照

Spinning Up 重点放在 Deep RL，不系统覆盖 tabular control。本章是进入 DQN 前的桥。DQN 的 TD target 可以直接看成 Q-learning target 的神经网络版本。

## 学完标准

- 能区分 MC、SARSA、Expected SARSA、Q-learning。
- 能解释 on-policy/off-policy。
- 能手写三个 TD control 更新式。
- 能跑通所有 tabular demo，并在报告中比较至少两种算法。
