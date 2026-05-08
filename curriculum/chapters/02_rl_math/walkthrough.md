# 02 代码逐行走读

本章逐行走读 `value_iteration_demo.py`，再定位到 `src/drl_lab/algorithms/tabular/value_iteration.py` 的核心 Bellman backup，以及 `gridworld.py` 的环境转移逻辑。

## 主线

```text
GridWorld -> value_iteration -> values -> policy -> reshape -> print
```

这里没有神经网络、loss、optimizer。学习重点是 state/action/reward/return/value/policy/Bellman/terminal/bootstrap。

## `value_iteration_demo.py` 逐行走读

`from __future__ import annotations` 是类型注解兼容设置，不是重点。

`from drl_lab.algorithms.tabular import GridWorld, value_iteration` 引入表格环境和 value iteration 算法。

`def main() -> None:` 定义脚本入口。

`env = GridWorld()` 创建默认 4x4 网格世界。

默认配置里有 16 个 state，动作是 `up/right/down/left`，终点是 state 15，每走一步 reward 是 `-1`，到终点 reward 是 `0`。

`values, policy = value_iteration(env, gamma=1.0)` 运行 value iteration。

`gamma=1.0` 表示不折扣未来 reward。因为这个小环境一定能到终点，所以先用 1.0 方便手算。

`values` 是 shape `[n_states]` 的数组，保存每个 state 的最优长期价值。

`policy` 是 shape `[n_states]` 的数组，保存每个 state 的贪心 action id。

`print("action legend: 0=up, 1=right, 2=down, 3=left")` 打印动作图例。读 policy 前必须先看它。

`print("values:")` 打印 value 标题。

`print(values.reshape(env.config.rows, env.config.cols))` 把长度 16 的 value 展成 4x4 矩阵，方便对应网格位置阅读。

`print("policy:")` 打印 policy 标题。

`print(policy.reshape(env.config.rows, env.config.cols))` 把长度 16 的 action id 展成 4x4 矩阵。

`if __name__ == "__main__": main()` 表示直接运行文件时执行 `main()`。

常见误解：`reshape` 只是改变显示形状，不会把算法从一维变成二维。算法内部 state 仍是 0 到 15。

## `gridworld.py` 环境转移逐段走读

`GridWorldConfig` 保存环境配置。

`rows=4`、`cols=4` 表示 4x4 网格。

`start_state=0` 是起点。

`terminal_state=15` 是终点。

`step_reward=-1.0` 表示普通一步扣 1 分。

`terminal_reward=0.0` 表示走到终点这一步给 0。

`GridWorld.actions = ("up", "right", "down", "left")` 定义 action id 到动作名字的顺序。这个顺序必须和 demo 的 action legend 对上。

`self.n_states = rows * cols` 得到 16 个 state。

`self.n_actions = len(self.actions)` 得到 4 个 action。

`reset` 把环境状态设回起点。

`step(action)` 对当前 state 执行动作，并更新 `self.state`。

`to_row_col(state)` 用 `divmod(state, cols)` 把 state id 转成行列。

`to_state(row, col)` 把行列转回 state id。

`is_terminal(state)` 判断是不是终点。

`transition(state, action)` 是 value iteration 最关心的函数：给定 state 和 action，返回 `next_state, reward, done`。

`if action < 0 or action >= self.n_actions` 防止非法动作。

`if self.is_terminal(state): return state, terminal_reward, True` 表示已经在终点时 episode 结束。

`row, col = self.to_row_col(state)` 取当前位置。

动作 0 用 `row = max(0, row - 1)`，表示向上但不能越过上边界。

动作 1 用 `col = min(cols - 1, col + 1)`，表示向右但不能越过右边界。

动作 2 用 `row = min(rows - 1, row + 1)`，表示向下但不能越过下边界。

动作 3 用 `col = max(0, col - 1)`，表示向左但不能越过左边界。

`next_state = self.to_state(row, col)` 把新行列转成 state id。

`done = self.is_terminal(next_state)` 判断是否到终点。

`reward = terminal_reward if done else step_reward` 到终点给 0，否则给 -1。

`return next_state, reward, done` 返回一步转移结果。

## `value_iteration.py` 核心逐行走读

`def value_iteration(env, gamma=0.99, theta=1e-8, max_iterations=10000)` 定义算法入口。

`gamma` 是折扣因子；`theta` 是停止阈值；`max_iterations` 是最大迭代轮数。

`if not 0.0 <= gamma <= 1.0` 防止非法折扣。

`if theta <= 0` 防止非法停止阈值。

`values = np.zeros(env.n_states, dtype=np.float64)` 初始化每个 state 的 value 为 0。

`policy = np.zeros(env.n_states, dtype=np.int64)` 初始化每个 state 的 action 为 0。

`for _ in range(max_iterations)` 开始反复扫表。

`delta = 0.0` 记录这一轮 value 最大变化量。

`for state in range(env.n_states)` 遍历每个 state。

`if env.is_terminal(state): continue` 跳过 terminal。学习点是：终点不需要继续更新 value。

`action_values = np.zeros(env.n_actions, dtype=np.float64)` 准备存四个 action 的 backup 结果。

`for action in range(env.n_actions)` 枚举 up、right、down、left。

`next_state, reward, done = env.transition(state, action)` 查询一步后会到哪、得到多少 reward、是否结束。

`bootstrap = 0.0 if done else gamma * values[next_state]` 是本章最关键的一行。done 时不看下一状态 value；没结束时才加折扣后的下一状态 value。

`action_values[action] = reward + bootstrap` 得到这个 action 的 Bellman backup。

`best_value = float(action_values.max())` 选四个 action 里最高的长期价值。

`delta = max(delta, abs(best_value - values[state]))` 记录这一轮变化幅度。

`values[state] = best_value` 更新 state value。

`policy[state] = int(action_values.argmax())` 记录让 value 最大的 action id。

`if delta < theta: break` 如果整轮变化很小，说明 value 基本收敛，提前停止。

`return values, policy` 返回最终 value 表和 greedy policy 表。

## 手算一个 backup

假设已经知道 `values[14] = 0`，现在算 state 13 向右：

```text
transition(13, right) -> next_state=14, reward=-1, done=False
bootstrap = 1.0 * values[14] = 0
action_value = -1 + 0 = -1
```

再算 state 14 向右：

```text
transition(14, right) -> next_state=15, reward=0, done=True
bootstrap = 0
action_value = 0
```

这解释了为什么越靠近终点，value 越高。

## 本章观察点

读输出时按顺序做：

1. 先读 action legend。
2. 再看 values 矩阵，确认靠近终点的数字更高。
3. 最后看 policy 矩阵，把数字翻译成动作。
4. 看到 terminal 位置的 policy 数字时，不要过度解读，因为 terminal state 在算法里被跳过。

## 公式对应

`value_iteration.py` 的核心四行可以直接对应 Bellman optimality equation：

```text
V*(s) = max_a [R(s,a,s') + gamma V*(s')]
```

代码对应：

```text
env.transition(state, action)              -> s', r, done
bootstrap = 0 if done else gamma * V[s']   -> terminal mask
action_values[action] = reward + bootstrap -> action backup
action_values.max()                        -> max over actions
```

policy extraction 对应：

```text
pi*(s) = argmax_a [R(s,a,s') + gamma V*(s')]
```

代码就是 `policy[state] = int(action_values.argmax())`。

## shape 对照表

| 对象 | shape | 含义 |
| --- | --- | --- |
| `values` | `[16]` | 每个 state 一个 value |
| `policy` | `[16]` | 每个 state 一个 action id |
| `values.reshape(4, 4)` | `[4, 4]` | 只为阅读方便 |
| `action_values` | `[4]` | 当前 state 下四个 action 的 backup |

公开课级读法：`reshape` 改的是显示方式，不改变 MDP 的 state 编号定义。
