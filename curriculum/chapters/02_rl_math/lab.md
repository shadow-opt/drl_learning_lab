# 02 实验说明书

## 实验目标

运行 GridWorld value iteration demo，并能解释 values 矩阵、policy 矩阵、action legend、terminal 和 Bellman backup。这个实验没有 PyTorch 训练，不会生成 checkpoint 或 ONNX。

## 运行命令

```bash
conda run -n drl-lab python curriculum/chapters/02_rl_math/code/value_iteration_demo.py
```

期望输出：

```text
action legend: 0=up, 1=right, 2=down, 3=left
values:
[[-5. -4. -3. -2.]
 [-4. -3. -2. -1.]
 [-3. -2. -1.  0.]
 [-2. -1.  0.  0.]]
policy:
[[1 1 1 2]
 [1 1 1 2]
 [1 1 1 2]
 [1 1 1 0]]
```

如果 NumPy 打印空格略有不同是正常的。核心是三部分都出现：action legend、values、policy。

## 逐项解释

- `action legend`：把 policy 数字翻译成动作。`1` 是 right，`2` 是 down。
- `values`：每个 state 的最优长期 return。因为每走一步扣 1，离终点越近通常越高。
- `policy`：每个 state 选择哪个 action。数字是动作编号，不是分数。
- terminal state 是 15，显示在右下角。算法跳过 terminal，所以它的 policy 数字不需要解释成真实动作。

## 怎么读 values

4x4 矩阵对应 state：

```text
0   1   2   3
4   5   6   7
8   9   10  11
12  13  14  15
```

例如 `values[0] = -5`，意思是从左上角按最优路线到终点前，未来累计 reward 是 -5。

为什么不是 -6：因为走到终点那一步 reward 是 0，普通步才是 -1。从 state 0 到 state 15 最短 6 步，其中最后一步进入终点给 0，所以累计是 5 个 `-1`。

## 怎么读 policy

如果第一行 policy 是：

```text
[1 1 1 2]
```

解释为：

- state 0：action 1，往右。
- state 1：action 1，往右。
- state 2：action 1，往右。
- state 3：action 2，往下。

常见误读：把 `2` 当成 value 2。它实际是 `down`。

## artifact

本章默认没有持久 artifact。它只在终端打印 values 和 policy。

如果你把输出重定向到文件，例如：

```bash
conda run -n drl-lab python curriculum/chapters/02_rl_math/code/value_iteration_demo.py > experiments/value_iteration_output.txt
```

这个文本文件可以作为学习记录 artifact，但不是代码默认生成的模型文件。

## 异常情况

输出没有 `action legend`：

- 先确认运行的是 `curriculum/chapters/02_rl_math/code/value_iteration_demo.py`。

policy 数字看不懂：

- 回到第一行 action legend。
- 记住 policy 数字是动作编号，不是价值。

values 不符合直觉：

- 检查 `gamma` 是否仍是 `1.0`。
- 检查 `step_reward` 是否仍是 `-1.0`，`terminal_reward` 是否仍是 `0.0`。
- 检查 terminal state 是否仍是 15。

把 terminal 继续 bootstrap：

- 如果你改代码让 done 时仍然加 `gamma * values[next_state]`，很多终点附近的手算会被污染。不要提交这种改动。

## 正常完成标准

你完成本实验，需要能回答：

1. reward 和 return 的区别。
2. state 14 向右为什么 backup 是 0。
3. state 13 向右为什么 backup 是 -1。
4. policy 矩阵里的 `1` 和 `2` 分别代表什么。
5. 为什么 terminal state 的 policy 数字不要过度解释。

## Ablation

这些 ablation 用于理解 Bellman backup，不要提交临时改动。

| 改动 | 预期现象 | 解释 |
| --- | --- | --- |
| `gamma=0.9` | 远离终点的 value 不再是简单步数负值 | 未来 reward 被折扣 |
| `step_reward=-0.1` | values 绝对值变小 | 每步惩罚变轻 |
| `terminal_reward=10` | 终点附近 values 升高 | 进入终点的 reward 直接进入 backup |
| 去掉 done mask | terminal 附近 target 被污染 | episode 结束后不应 bootstrap |

## 诊断决策树

```text
values 全是 0？
  查是否跳过了所有 non-terminal state 或 max_iterations 太小
policy 数字不懂？
  回到 action legend
state 14 backup 算错？
  查 terminal_reward 和 done mask
边界行动不符合直觉？
  查 gridworld.py 的 max/min 边界裁剪
gamma 改变后 values 变了？
  这是 return 定义改变，不是 bug
```

## 记录表

| State/action | next_state | reward | done | backup |
| --- | --- | --- | --- | --- |
| 14/right | | | | |
| 13/right | | | | |
| 10/right | | | | |
| 3/down | | | | |
