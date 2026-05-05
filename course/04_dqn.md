# 04 DQN

## 学习目标

DQN 是从 tabular Q-learning 进入 Deep RL 的第一座桥。学完本章后，你应该能把 tabular Q-learning 的 update rule 改写成神经网络 loss，解释 replay buffer、target network、epsilon-greedy、Huber loss、gradient clipping、checkpoint、eval 和 Q-network ONNX export 的作用。

更具体地说，你要能回答三个问题：第一，为什么 Q-learning 可以离线重用旧数据；第二，为什么 DQN 不能直接用同一个网络同时预测当前值和目标值；第三，导出 Q-network 时为什么输入只有 observation，输出是每个离散 action 的 Q-value。

## 为什么重要

前面的 tabular Q-learning 用表格保存 `Q(s,a)`。当 state 很多、observation 是连续向量或图像时，表格无法覆盖所有可能输入。DQN 的核心替换是：

$$
Q_\theta(s,a) \approx Q^*(s,a)
$$

也就是用参数为 `theta` 的神经网络近似最优动作价值函数。这个替换看起来简单，但训练性质变了：Q target 依赖网络自己的预测，采样数据又来自当前行为策略，样本之间高度相关。如果直接在线更新，很容易振荡或发散。DQN 的 replay buffer 和 target network 就是两个关键稳定器。

DQN 也是后面 off-policy actor-critic 的工程预备课。DDPG、TD3、SAC 都会继续使用 replay buffer、target networks、TD target、checkpoint、eval 和导出验证。

## 核心直觉

先把 tabular Q-learning 写成一句话：当前 `(s,a)` 的 Q 值，应该接近“一步 reward 加上下一个 state 能拿到的最好 Q 值”。DQN 只是把“更新表格里的一个格子”变成“让神经网络对这个 `(s,a)` 的输出靠近 target”。

Replay buffer 的直觉是把环境交互样本存起来，训练时随机抽 batch。这样一方面提高样本利用率，另一方面打散连续时间步之间的相关性。Target network 的直觉是给 TD target 一个慢变化的参照物。如果 target 每个 gradient step 都跟着 online network 变，优化目标会漂移得太快。

Epsilon-greedy 是 DQN 的探索机制：大多数时候选当前 Q 最大的 action，少数时候随机探索。训练早期 epsilon 高，后期逐渐降低。Eval 时通常关闭或显著降低探索，否则评估结果混入随机行为。

## 数学与公式

Tabular Q-learning 更新是：

$$
Q(s,a) \leftarrow Q(s,a) + \alpha
\left[r + \gamma(1-done)\max_{a'}Q(s',a') - Q(s,a)\right]
$$

DQN 把中括号里的目标写成：

$$
y = r + \gamma (1-done)\max_{a'}Q_{\theta^-}(s',a')
$$

这里 `theta^-` 是 target network 参数。Online network 给出当前 batch 的全部 action value：

$$
Q_\theta(s) = [Q_\theta(s,a_1), Q_\theta(s,a_2), \ldots]
$$

因为 batch 中每条 transition 只执行了一个 action，所以代码里用 `gather` 取出对应 action 的值：

```python
q_values = q_network(batch.obs)
chosen_q = q_values.gather(dim=1, index=batch.actions.unsqueeze(1)).squeeze(1)
```

最终 loss 是：

$$
L(\theta)=\mathrm{Huber}(Q_\theta(s,a), y)
$$

本仓库在 `src/drl_lab/algorithms/dqn/losses.py` 使用 `F.smooth_l1_loss`，也就是 Huber loss。它比 MSE 对异常 TD error 更稳。

Shape 约定：

```text
obs:       [batch, obs_dim]
actions:   [batch]
rewards:   [batch]
dones:     [batch]
q_values:  [batch, action_dim]
chosen_q:  [batch]
target:    [batch]
```

## 算法流程

1. 初始化 online Q-network 和 target Q-network。
2. 将 target network 同步为 online network 参数。
3. 用 epsilon-greedy policy 和环境交互。
4. 将 `(obs, action, reward, next_obs, done)` 写入 replay buffer。
5. 当 buffer 达到 `learning_starts` 后，随机采样 batch。
6. 用 target network 在 `torch.no_grad()` 下计算 TD target。
7. 用 online network 计算 `chosen_q`。
8. 用 Huber loss 更新 online network。
9. 按固定间隔同步 target network。
10. 周期性 eval、保存 checkpoint、导出 ONNX。

注意：target 计算必须停止梯度。否则网络会一边追目标，一边移动目标，loss 的含义会被破坏。

## 代码双索引

- 教学 demo：`labs/04_dqn/code/dqn_smoke_demo.py`
- Replay buffer：`src/drl_lab/algorithms/dqn/buffer.py`
- Q-network：`src/drl_lab/algorithms/dqn/networks.py`
- Loss：`src/drl_lab/algorithms/dqn/losses.py`
- Agent/update：`src/drl_lab/algorithms/dqn/agent.py`
- 训练入口：`src/drl_lab/algorithms/dqn/train.py`
- Eval：`src/drl_lab/algorithms/dqn/eval.py`
- ONNX export：`src/drl_lab/algorithms/dqn/export_onnx.py`
- 测试：`tests/test_dqn.py`

阅读顺序建议：先看 `losses.py`，确认公式；再看 `buffer.py`，确认 batch shape；最后看 `train.py`，把环境交互、更新、eval、checkpoint、export 串起来。

## 实验任务

先跑 smoke demo：

```bash
conda run -n drl-lab python labs/04_dqn/code/dqn_smoke_demo.py
```

再跑短训练：

```bash
conda run -n drl-lab python -m drl_lab.algorithms.dqn.train --total-steps 300 --learning-starts 32
```

观察：

- replay buffer 是否先填充再训练；
- loss 是否 finite；
- episode return 是否有改善迹象；
- epsilon 是否按计划变化；
- run 目录是否包含 metrics、checkpoint 和 ONNX。

然后完成 `labs/04_dqn/exercises.md`，在 `labs/04_dqn/report.md` 写出 DQN target 和一次失败记录。

## Debug Checklist

- `batch.actions` 是否是 integer index，shape 为 `[batch]`。
- `gather` 的维度是否正确。
- target network 是否在 `no_grad` 下计算。
- terminal transition 是否使用 `(1 - done)` 去掉 bootstrap。
- replay buffer 采样前是否已有足够 transition。
- online/target network 是否按预期同步，而不是每步都混淆。
- eval 是否关闭 epsilon 探索。
- ONNX 导出是否只包含 Q-network inference。

## Spinning Up 对照

Spinning Up 不覆盖 DQN。本章是本仓库为 Deep RL 入门额外补齐的 value-based 算法。它的位置在 tabular Q-learning 之后、policy gradient 之前：你先学会如何把 Bellman target 写成神经网络 loss，再学习如何直接优化 policy。

DQN 的工程模式会继续出现在 DDPG、TD3、SAC 中：replay buffer、target network、TD target、off-policy batch update、eval 和导出。因此本章不是支线，而是进入现代 off-policy DRL 的基础。

## 学完标准

- 能从 tabular Q-learning 更新式推到 DQN TD target。
- 能解释 replay buffer 和 target network 分别稳定什么。
- 能说明 `obs/actions/rewards/dones/q_values/target` 的 shape。
- 能跑通 smoke demo 和短训练。
- 能在报告中记录 loss、return、epsilon、checkpoint 和 ONNX artifact。
- 能解释 Q-network ONNX 输入输出，并通过相关测试。
