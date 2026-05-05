# VPG

## 学习目标

VPG，Vanilla Policy Gradient，是最直接的 policy-gradient 算法。学完本章后，你应该能解释 trajectory、return-to-go、log probability、advantage、value baseline、policy loss 和 value loss，并能从“提高好 action 的概率”推到代码里的 `-(log_prob * advantage).mean()`。

你还要能区分 value-based 和 policy-based 方法：DQN 先学 `Q(s,a)` 再选最大 action；VPG 直接学习 `pi(a|s)`，用采样得到的回报告诉 policy 哪些 action 更应该出现。

## 为什么重要

VPG 是 PPO、TRPO 和很多 actor-critic 方法的起点。如果 VPG 的 `log_prob * advantage` 没理解，PPO 的 ratio、TRPO 的 surrogate objective、SAC 的 stochastic actor 都会变成公式堆叠。

VPG 也暴露了 policy-gradient 的核心困难：采样方差很大、on-policy 数据不能随便重用、训练效率低。后面的 PPO 和 TRPO 主要就是在解决“policy 更新太不稳定”和“采样数据太贵”的问题。

## 核心直觉

想象一次 episode 中，policy 在某个 state 选了一个 action。后来发现这次选择带来的结果比预期好，那就应该提高这个 action 在这个 state 下的概率；如果结果比预期差，就降低概率。

神经网络 policy 输出的是一个概率分布。我们不能直接对“采样 action”求导，但可以对这个 action 的 log probability 求导。`log_prob` 告诉我们：当前 policy 有多倾向于做出这个 action。把它乘以 advantage，就得到“这次倾向应该被加强还是削弱”。

Value baseline 的作用是降低方差。我们不只问“这次 return 是多少”，而是问“这次 action 比这个 state 的平均水平好多少”。这就是 advantage：

$$
A(s,a)=Q(s,a)-V(s)
$$

## 数学与公式

目标是最大化 expected return：

$$
J(\theta)=\mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]
$$

Policy gradient theorem 给出：

$$
\nabla_\theta J(\pi_\theta)=
\mathbb{E}\left[\nabla_\theta \log \pi_\theta(a_t|s_t) A^\pi(s_t,a_t)\right]
$$

实现中通常最小化负号形式：

$$
L_\pi(\theta)=-\mathbb{E}[\log \pi_\theta(a_t|s_t)\hat{A}_t]
$$

本仓库对应代码在 `src/drl_lab/algorithms/vpg/losses.py`：

```python
log_probs = policy.log_prob(obs, actions)
return -(log_probs * advantages).mean()
```

Value function 用 return target 做 MSE：

$$
L_V(\phi)=\mathbb{E}[(V_\phi(s_t)-\hat{R}_t)^2]
$$

Shape 约定：

```text
obs:         [batch, obs_dim]
actions:     [batch]
advantages:  [batch]
returns:     [batch]
log_probs:   [batch]
values:      [batch]
```

## 手算样例：为什么是 `-log_prob * advantage`

假设一个 batch 里只有两个 timestep：

```text
t0: action=left,  log_prob=-0.20, advantage=+2.0
t1: action=right, log_prob=-1.00, advantage=-1.0
```

policy objective 希望最大化 `log_prob * advantage`：

```text
t0: -0.20 * +2.0 = -0.40
t1: -1.00 * -1.0 = +1.00
mean = 0.30
```

优化器默认最小化 loss，所以代码取负号：

```text
policy_loss = -0.30
```

符号怎么理解：

- `advantage > 0`：这个 action 比 baseline 好，梯度会提高它的 log probability。
- `advantage < 0`：这个 action 比 baseline 差，梯度会降低它的 log probability。

不要把 `log_prob` 的数值正负当成“好坏”。离散概率的 log 通常小于等于 0，真正决定方向的是 advantage 的符号和外层负号。

## 算法流程

1. 用当前 policy 采样一批 trajectories。
2. 对每个 timestep 计算 return-to-go。
3. 用 value function 估计 baseline。
4. 计算并标准化 advantage。
5. 用 `-log_prob * advantage` 更新 policy。
6. 用 MSE 更新 value function。
7. 评估 policy，并保存 checkpoint/export。

因为 VPG 是 on-policy，采样数据只适合当前 policy。policy 更新后，旧 trajectory 的分布就不再匹配新 policy，不能像 DQN replay buffer 那样无限重用。

## 代码双索引

- Demo：`labs/05_policy_gradient/vpg/code/vpg_core_demo.py`
- Policy/value networks：`src/drl_lab/algorithms/vpg/networks.py`
- Buffer：`src/drl_lab/algorithms/vpg/buffer.py`
- Loss：`src/drl_lab/algorithms/vpg/losses.py`
- Agent：`src/drl_lab/algorithms/vpg/agent.py`
- Train：`src/drl_lab/algorithms/vpg/train.py`
- Eval：`src/drl_lab/algorithms/vpg/eval.py`
- 测试：`tests/test_vpg.py`

建议先看 `losses.py`，再看 `buffer.py` 中 return 和 advantage 的构造，最后看 `train.py` 如何收集 rollout。

## 实验任务

```bash
conda run -n drl-lab python labs/05_policy_gradient/vpg/code/vpg_core_demo.py
```

短 CartPole smoke train：

```bash
conda run -n drl-lab python -m drl_lab.algorithms.vpg.train --epochs 1 --steps-per-epoch 64
```

这是链路验证，不是收敛训练。观察：

- policy loss 是否 finite；
- value loss 是否下降；
- advantage 标准化后是否接近均值 0；
- policy/value 是否成功导出 ONNX；
- eval return 是否有改善迹象。
- run 目录是否生成 `config.json`、`environment.json`、`metrics.csv`、`policy.pt`、`policy.onnx`、`eval_result.json`、`export_report.json`。

在 `labs/05_policy_gradient/vpg/report.md` 中写出你对 `log_prob * advantage` 的解释。

## Debug Checklist

- `actions`、`advantages`、`returns` 是否都是 `[batch]`。
- `log_prob` 是否来自采样时执行的 action。
- policy loss 符号是否正确：实现里是负号，因为 optimizer 做最小化。
- advantage 是否标准化，且没有跨错误 episode 边界计算。
- value target 是否是 return，而不是 advantage。
- on-policy rollout 是否被错误重用太多次。
- eval 时是否使用 deterministic 或 greedy action。

## Spinning Up 对照

Spinning Up 的 VPG 章节是本算法主参考。它讲清了 policy gradient、reward-to-go 和 baseline。本仓库保留这条概念主线，但额外要求：

- 明确 tensor shape；
- 用现代 PyTorch `nn.Module` 组织 policy/value；
- 用 tests 验证 loss、buffer 和训练 smoke；
- 保存 checkpoint 和 eval artifact；
- 导出 policy/value 并做 ONNXRuntime 一致性检查。

## 学完标准

- 能解释为什么 advantage 为正会提高 action 概率。
- 能写出 VPG policy loss 和 value loss。
- 能说明 VPG 为什么是 on-policy。
- 能跑通 VPG demo 和短训练。
- 能在源码中指出 policy loss、value loss、advantage normalization 的位置。
- 能完成 `labs/05_policy_gradient/vpg/report.md`。
