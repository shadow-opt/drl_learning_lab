# PPO

## 学习目标

PPO，Proximal Policy Optimization，是 VPG 之后最重要的 on-policy baseline。学完本章后，你应该能解释 old policy、new policy、log-prob ratio、clipped surrogate objective、GAE、value loss、entropy bonus、approximate KL 和 early stopping。

最关键的目标是理解：PPO 不是换了一个完全不同的 RL 框架，而是在 VPG 的 policy-gradient objective 上加了“不要一步更新太远”的约束。

## 为什么重要

VPG 的问题是每次 policy update 都可能过大。因为采样数据来自旧 policy，如果新 policy 离旧 policy 太远，用旧数据估计新 policy 的效果就会变得不可靠。TRPO 用显式 KL 约束处理这个问题，但实现复杂。PPO 用 clipping 做了一个更容易实现、工程上更常用的近似。

PPO 是很多复杂系统中的默认起点。即使本仓库当前不进入机器人或大型仿真，理解 PPO 也能为以后阅读 legged locomotion、RLHF 或连续控制代码打基础。

## 核心直觉

PPO 会比较同一批 rollout 中，旧 policy 和新 policy 对同一个 action 的概率。这个比较由 ratio 表示：

```text
ratio > 1：新 policy 比旧 policy 更倾向这个 action
ratio < 1：新 policy 比旧 policy 更不倾向这个 action
```

如果 advantage 为正，说明这个 action 比平均水平好，ratio 变大是好事；但不能无限变大，否则 policy 更新过猛。如果 advantage 为负，说明这个 action 不好，ratio 变小是好事；但也不能无限变小。Clipping 就是在这两个方向上截断收益。

## 数学与公式

概率比：

$$
r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
$$

实际代码通常用 log prob 差值计算，数值更稳定：

$$
r_t(\theta)=\exp(\log\pi_\theta(a_t|s_t)-\log\pi_{\theta_{old}}(a_t|s_t))
$$

PPO clipped objective：

$$
L^{CLIP}(\theta)=
\mathbb{E}\left[
\min(r_t\hat{A}_t,
\mathrm{clip}(r_t,1-\epsilon,1+\epsilon)\hat{A}_t)
\right]
$$

本仓库优化器做最小化，所以代码返回负号形式。对应 `src/drl_lab/algorithms/ppo/losses.py`：

```python
ratio = torch.exp(new_log_probs - old_log_probs)
unclipped = ratio * advantages
clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
loss = -torch.min(unclipped, clipped).mean()
```

Approximate KL 诊断：

$$
\widehat{KL} \approx \mathbb{E}[\log\pi_{old}(a|s)-\log\pi_{new}(a|s)]
$$

Shape 约定：

```text
obs:            [batch, obs_dim]
actions:        [batch]
advantages:     [batch]
old_log_probs:  [batch]
ratio:          [batch]
returns:        [batch]
```

## 手算样例：PPO clipping 怎么管住正负 advantage

设 `clip_ratio=0.2`，所以 ratio 会被截到 `[0.8, 1.2]`。PPO 对每个样本取：

```text
min(ratio * advantage, clipped_ratio * advantage)
```

几个典型情况：

| advantage | ratio | unclipped | clipped | PPO 取值 | 直觉 |
| --- | ---: | ---: | ---: | ---: | --- |
| +1.0 | 1.5 | 1.5 | 1.2 | 1.2 | 好 action 概率不能涨太猛 |
| +1.0 | 0.7 | 0.7 | 0.8 | 0.7 | 概率变低不是收益，不阻止 |
| -1.0 | 0.5 | -0.5 | -0.8 | -0.8 | 坏 action 概率不能降太猛 |
| -1.0 | 1.3 | -1.3 | -1.2 | -1.3 | 概率变高更差，不阻止 |

关键点：`min` 在 advantage 为负时看起来反直觉，因为乘了负数后大小关系会翻转。不要只背公式，要把正负 advantage 分开看。

## 算法流程

1. 用当前 policy 采样 rollout。
2. 保存每个 action 的 `old_log_prob`。
3. 计算 return 和 GAE advantage。
4. 将 rollout batch 切成 mini-batches。
5. 对同一批 rollout 做多个 epoch 更新。
6. 每次更新计算 new log prob、ratio、clipped loss。
7. 同时更新 value function，并可加入 entropy bonus。
8. 监控 approximate KL；过大时停止 policy update。
9. 周期性 eval、checkpoint 和 ONNX export。

PPO 可以多次使用同一批 rollout，但仍然是 on-policy。它不是 DQN 那种长期 replay；旧数据只能在短时间内、受约束地复用。

## 代码双索引

- Demo：`curriculum/chapters/05_policy_gradient/ppo/code/ppo_core_demo.py`
- Buffer/GAE：`src/drl_lab/algorithms/ppo/buffer.py`
- Loss：`src/drl_lab/algorithms/ppo/losses.py`
- Agent：`src/drl_lab/algorithms/ppo/agent.py`
- Train：`src/drl_lab/algorithms/ppo/train.py`
- Eval：`src/drl_lab/algorithms/ppo/eval.py`
- Policy/value networks：复用 `src/drl_lab/algorithms/vpg/networks.py`
- 测试：`tests/test_ppo.py`

阅读顺序：先看 `losses.py`，确认 clipping；再看 `buffer.py`，确认 GAE；最后看 `agent.py` 或 `train.py`，理解多 epoch update。

## 实验任务

```bash
conda run -n drl-lab python curriculum/chapters/05_policy_gradient/ppo/code/ppo_core_demo.py
```

短 CartPole smoke train：

```bash
conda run -n drl-lab python -m drl_lab.algorithms.ppo.train --epochs 1 --steps-per-epoch 64
```

这是链路验证，不是收敛训练。观察：

- ratio 是否围绕 1；
- clip fraction 是否过大；
- approx KL 是否异常增大；
- entropy 是否过早塌缩；
- policy loss 和 value loss 是否 finite；
- policy/value ONNX 是否导出成功。
- run 目录是否生成 `config.json`、`environment.json`、`metrics.csv`、`policy.pt`、`policy.onnx`、`eval_result.json`、`export_report.json`。

在 `curriculum/chapters/05_policy_gradient/ppo/report.md` 中解释 clipping 对正 advantage 和负 advantage 分别做了什么。

## Debug Checklist

- old log prob 是否在 rollout 采样时保存，而不是更新后重新算。
- ratio 是否用 `exp(new_logp - old_logp)`。
- advantage 是否标准化。
- clipped objective 的 `min` 是否处理了正负 advantage。
- value target 是否和 GAE/return 计算一致。
- 多 epoch 更新是否过多，导致 KL 过大。
- entropy bonus 系数是否让 policy 过早确定。
- eval 是否和训练采样逻辑分开。

## Spinning Up 对照

Spinning Up PPO 是本章核心参考。阅读时重点看三件事：clipped surrogate objective、GAE、KL early stopping。本仓库在其基础上补充：

- 更明确的 shape 检查；
- `old_log_probs` 的数据流；
- 单元测试覆盖 ratio、clip 和 diagnostics；
- checkpoint、eval、ONNX export；
- 与 VPG 共享 policy/value 网络，便于比较算法差异。

## 学完标准

- 能从 VPG loss 推出 PPO ratio 形式。
- 能解释 clipped objective 如何限制 policy 更新。
- 能说明 PPO 为什么仍然是 on-policy。
- 能在源码中指出 ratio、clip、approx KL、GAE 的位置。
- 能跑通 PPO core demo 和短训练。
- 能完成 `curriculum/chapters/05_policy_gradient/ppo/report.md`。
