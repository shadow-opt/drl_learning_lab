# DDPG

## 学习目标

DDPG，Deep Deterministic Policy Gradient，是连续动作空间中的 off-policy actor-critic。学完本章后，你应该能解释 deterministic actor、continuous critic、replay buffer、target actor、target critic、soft update、exploration noise、actor loss、critic loss，以及 actor/critic 的 ONNX 导出。

你还要能区分 DQN 和 DDPG：DQN 输出每个离散 action 的 Q 值，然后取最大；DDPG 让 actor 直接输出连续 action，再用 critic 评价这个 action。

## 为什么重要

很多控制任务的 action 不是离散按钮，而是连续数值，比如力矩、速度、角度。DQN 无法枚举所有连续 action。DDPG 的做法是同时学习两个函数：

$$
\mu_\theta(s): s \rightarrow a
$$

$$
Q_\phi(s,a): (s,a) \rightarrow q
$$

Actor `mu` 负责给出动作，critic `Q` 负责评价动作。Actor 的训练目标就是让 critic 对它输出的 action 打更高分。

DDPG 也是 TD3 和 SAC 的起点。TD3 修正 DDPG 的过估计和不稳定，SAC 则把 deterministic actor 改成 maximum entropy stochastic actor。

## 核心直觉

可以把 DDPG 想成“DQN + 一个会搜索连续 action 的 actor”。在 DQN 中，`max_a Q(s,a)` 可以通过枚举离散 action 完成；在连续动作中无法枚举，所以 actor 学一个近似的 argmax：

$$
\mu_\theta(s) \approx \arg\max_a Q_\phi(s,a)
$$

Critic 仍然通过 Bellman target 学习。Actor 不直接看 reward，而是通过 critic 的梯度知道“往哪个 action 方向能让 Q 更大”。因为 deterministic actor 本身不会随机探索，训练时必须额外加 exploration noise。

## 数学与公式

Critic target：

$$
y = r + \gamma(1-done)Q_{\phi^-}(s',\mu_{\theta^-}(s'))
$$

Critic loss：

$$
L_Q(\phi)=\mathbb{E}[(Q_\phi(s,a)-y)^2]
$$

Actor objective：

$$
J(\theta)=\mathbb{E}[Q_\phi(s,\mu_\theta(s))]
$$

因为 PyTorch optimizer 最小化 loss，代码写成：

$$
L_\mu(\theta)=-\mathbb{E}[Q_\phi(s,\mu_\theta(s))]
$$

本仓库对应 `src/drl_lab/algorithms/ddpg/losses.py`：

```python
next_actions = target_actor(batch.next_obs)
next_q = target_critic(batch.next_obs, next_actions)
target = batch.rewards + gamma * (1.0 - batch.dones) * next_q
```

Actor loss：

```python
actions = actor(obs)
loss = -critic(obs, actions).mean()
```

Shape 约定：

```text
obs:       [batch, obs_dim]
actions:   [batch, action_dim]
rewards:   [batch]
dones:     [batch]
q_values:  [batch]
```

## 算法流程

1. 初始化 actor、critic、target actor、target critic。
2. 将 target networks 同步为 online networks。
3. 用 actor 输出 action，并加入探索噪声。
4. 与环境交互，将 transition 写入 continuous replay buffer。
5. 随机采样 batch。
6. 用 target actor/critic 构造 critic target。
7. 更新 critic。
8. 用 `-critic(obs, actor(obs)).mean()` 更新 actor。
9. 对 target networks 做 Polyak soft update。
10. 周期性 eval、checkpoint、导出 actor 和 critic。

Soft update：

$$
\theta^- \leftarrow (1-\tau)\theta^- + \tau\theta
$$

`tau` 越小，target network 变化越慢。

## 代码双索引

- Demo：`curriculum/chapters/05_policy_gradient/ddpg/code/ddpg_core_demo.py`
- Actor/critic networks：`src/drl_lab/algorithms/ddpg/networks.py`
- Replay buffer：`src/drl_lab/algorithms/ddpg/buffer.py`
- Loss/soft update：`src/drl_lab/algorithms/ddpg/losses.py`
- Agent：`src/drl_lab/algorithms/ddpg/agent.py`
- Train：`src/drl_lab/algorithms/ddpg/train.py`
- Eval：`src/drl_lab/algorithms/ddpg/eval.py`
- 测试：`tests/test_ddpg.py`

先读 `losses.py`，再读 `networks.py` 的 actor action limit 和 critic multi-input shape，最后看 `train.py` 的探索噪声与导出。

## 实验任务

```bash
conda run -n drl-lab python curriculum/chapters/05_policy_gradient/ddpg/code/ddpg_core_demo.py
conda run -n drl-lab python -m drl_lab.algorithms.ddpg.train --total-steps 300 --learning-starts 32 --eval-episodes 1
```

观察：

- actor 输出是否在 action limit 内；
- critic loss 是否 finite；
- actor loss 是否是负的 critic score；
- target networks 是否 soft update；
- actor 和 critic ONNX 是否生成并通过一致性检查。

## Debug Checklist

- action shape 是否是 `[batch, action_dim]`。
- actor 输出是否按环境 action range 缩放。
- critic 是否同时接收 `obs` 和 `actions`。
- target critic 计算是否在 `torch.no_grad()` 内。
- actor loss 是否写成负号。
- replay buffer 是否存储 continuous action，而不是 discrete index。
- eval 是否关闭 exploration noise。
- critic ONNX 是否使用 multi-input named inputs。

## Spinning Up 对照

Spinning Up DDPG 是本章主参考。阅读重点是 deterministic actor-critic、target networks、replay buffer 和 Polyak averaging。本仓库在其基础上补充：

- 现代 PyTorch module 组织；
- actor/critic shape 检查；
- critic multi-input ONNX export；
- checkpoint、eval、metrics 和短训练 smoke test；
- 与 TD3/SAC 共享连续控制工程习惯。

## 学完标准

- 能解释 DDPG 为什么需要 actor 和 critic。
- 能写出 critic target、critic loss、actor loss。
- 能说明 exploration noise 为什么只用于训练交互。
- 能指出 soft update 的代码位置。
- 能跑通 DDPG demo 和短训练。
- 能解释 actor/critic ONNX 输入输出。
