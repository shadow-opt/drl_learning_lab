# SAC

## 学习目标

SAC，Soft Actor-Critic，是 maximum entropy off-policy actor-critic。学完本章后，你应该能解释 stochastic actor、squashed Gaussian policy、reparameterization trick、entropy bonus、temperature `alpha`、twin critics、soft target，以及训练采样路径和导出 deterministic 路径的区别。

SAC 是本仓库经典 DRL 主线中最现代、也最容易把工程细节写错的算法之一。它把 TD3 的 twin critics 和 off-policy replay 保留下来，同时让 policy 学会“高回报 + 足够随机”。

## 为什么重要

DDPG 和 TD3 使用 deterministic actor，探索依赖外部噪声。SAC 直接学习 stochastic policy，并把 entropy 写进目标函数。这样 policy 不只是追求高 Q，也会保持一定不确定性，减少过早收敛到坏策略的风险。

SAC 还训练 temperature `alpha` 来控制 entropy 强度。`alpha` 太大，policy 过于随机；太小，policy 过早确定。自动调节 alpha 是 SAC 工程实现中的关键环节。

## 核心直觉

SAC 的 actor 输出一个 Gaussian distribution，而不是一个固定 action。采样后通过 `tanh` 把 action 压到合法范围。训练时使用随机采样 action，因为 entropy 是目标的一部分；导出和 eval 时通常使用 deterministic mean action，因为部署需要稳定输出。

Critic target 不再只是未来 Q，还要减去 `alpha * log_prob`：

```text
高 Q 好，低 log_prob 代表低概率/低熵，需要 entropy 项调节
```

在实现中，actor loss 是：

```text
alpha * log_prob - min(Q1, Q2)
```

最小化这个 loss，相当于同时提高 Q 并保持 entropy。

## 数学与公式

Maximum entropy objective：

$$
J(\pi)=\sum_t \mathbb{E}[r(s_t,a_t)+\alpha \mathcal{H}(\pi(\cdot|s_t))]
$$

SAC critic target：

$$
y = r + \gamma(1-done)
\left(\min_i Q_{\phi_i^-}(s',a') - \alpha \log \pi(a'|s')\right)
$$

Actor loss：

$$
L_\pi=\mathbb{E}\left[\alpha \log \pi(a|s)-\min_i Q_{\phi_i}(s,a)\right]
$$

Temperature loss：

$$
L_\alpha = -\log\alpha \left(\log\pi(a|s) + \mathcal{H}_{target}\right)
$$

本仓库对应 `src/drl_lab/algorithms/sac/losses.py`：

```python
actions, log_probs = actor.sample(obs)
q1, q2 = critics(obs, actions)
min_q = torch.minimum(q1, q2)
loss = (alpha * log_probs - min_q).mean()
```

Squashed Gaussian actor 在 `src/drl_lab/algorithms/sac/networks.py` 中实现。`sample()` 返回 stochastic action 和 corrected log prob；`forward()` 返回 deterministic mean action，用于 inference/export。

Shape 约定：

```text
obs:        [batch, obs_dim]
actions:    [batch, action_dim]
log_probs:  [batch]
q1/q2:      [batch]
alpha:      scalar
```

## 算法流程

1. 初始化 squashed Gaussian actor、twin critics、target critics。
2. 用 actor.sample 与环境交互。
3. transition 存入 replay buffer。
4. 采样 batch。
5. 用 target actor.sample 生成 next action 和 next log prob。
6. 用 twin target critics 和 entropy term 构造 critic target。
7. 更新 twin critics。
8. 用 reparameterized sampled action 更新 actor。
9. 更新 temperature `alpha`。
10. Soft update target critics。
11. Eval/export 时使用 actor.forward 的 deterministic action。

## 代码双索引

- Demo：`curriculum/chapters/05_policy_gradient/sac/code/sac_core_demo.py`
- Actor network：`src/drl_lab/algorithms/sac/networks.py`
- Losses：`src/drl_lab/algorithms/sac/losses.py`
- Agent：`src/drl_lab/algorithms/sac/agent.py`
- Train：`src/drl_lab/algorithms/sac/train.py`
- Eval：`src/drl_lab/algorithms/sac/eval.py`
- Twin critics 复用：`src/drl_lab/algorithms/td3/networks.py`
- Replay buffer 复用：`src/drl_lab/algorithms/ddpg/buffer.py`
- 测试：`tests/test_sac.py`

阅读顺序建议：先看 `networks.py` 的 `sample()` 和 `forward()`，再看 `losses.py`，最后看 `agent.py` 中 alpha update 和 target update。

## 实验任务

```bash
conda run -n drl-lab python curriculum/chapters/05_policy_gradient/sac/code/sac_core_demo.py
conda run -n drl-lab python -m drl_lab.algorithms.sac.train --total-steps 300 --learning-starts 32 --eval-episodes 1
```

观察：

- actor sample 的 action 和 log_prob shape；
- critic loss、actor loss、alpha loss 是否 finite；
- alpha 是否有变化；
- eval 是否使用 deterministic action；
- actor 和 twin critics ONNX 是否导出成功。

## Debug Checklist

- `sample()` 是否使用 reparameterization，也就是 `rsample()`。
- tanh-squash 后 log prob 是否有 correction。
- `log_probs` 是否是 `[batch]`。
- critic target 是否使用 `min(Q1,Q2) - alpha * log_prob`。
- actor loss 是否是 `alpha * log_prob - min_q`。
- alpha loss 中 log prob 是否 detach。
- 导出 actor 时是否使用 `forward()`，避免随机采样。
- multi-input critic export 是否命名 `obs/actions`。

## Spinning Up 对照

Spinning Up SAC 是本章主参考。阅读时重点看 entropy-regularized objective、squashed Gaussian policy、twin Q-functions 和 alpha。需要注意的是，SAC 不只是“TD3 加 entropy”，它的 actor 是 stochastic policy，训练和导出路径必须分开。

本仓库相对 Spinning Up 的补充：

- 明确 `sample()` 和 `forward()` 的职责；
- 测试 tanh correction、loss shape、short training artifact；
- 导出 deterministic actor 和 twin critics；
- 用 ONNXRuntime 验证多输入 critic 一致性。

## 学完标准

- 能解释 maximum entropy objective。
- 能写出 SAC critic target、actor loss、temperature loss。
- 能说明 tanh log prob correction 为什么必要。
- 能区分训练采样路径和导出 deterministic 路径。
- 能跑通 SAC demo 和短训练。
- 能完成 `curriculum/chapters/05_policy_gradient/sac/report.md`。
