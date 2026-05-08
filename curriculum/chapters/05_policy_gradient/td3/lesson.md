# TD3

## 学习目标

TD3，Twin Delayed DDPG，是 DDPG 的稳定化版本。学完本章后，你应该能解释 clipped double Q-learning、twin critics、target policy smoothing、delayed actor update、soft update，以及它们分别解决 DDPG 的什么问题。

你不需要把 TD3 当成全新框架。它仍然是 deterministic off-policy actor-critic，只是在 critic target 和 actor update 节奏上更谨慎。

## 为什么重要

DDPG 的 critic 如果高估某些 action，actor 会被吸引到这些虚高区域，进一步放大错误。连续动作空间中这种问题很常见，因为 actor 可以沿着 critic 的误差方向优化。TD3 主要修三件事：

1. 用两个 critic，target 取较小值，降低过估计。
2. 给 target action 加小噪声，平滑 critic target。
3. 延迟 actor update，让 critic 先学得更稳。

这三项 trick 很实用，也能帮助你理解 SAC 为什么继续使用 twin critics。

## 核心直觉

Twin critics 像两个独立评委。如果一个 critic 对某个 action 过度乐观，取 `min(Q1, Q2)` 可以保守一点。Target policy smoothing 的直觉是：不要只让 critic 适应一个非常尖锐的 action，而是让它在 action 附近也保持平滑。Delayed actor update 的直觉是：actor 依赖 critic 的梯度，如果 critic 还没学稳，actor 更新越勤快越容易追错方向。

## 数学与公式

TD3 target action：

$$
\tilde{a}' = \mathrm{clip}(\mu_{\theta^-}(s') + \epsilon)
$$

其中噪声 `epsilon` 会先裁剪，再把 action 裁剪到合法范围。

TD target：

$$
y = r + \gamma(1-done)\min_{i=1,2} Q_{\phi_i^-}(s',\tilde{a}')
$$

Critic loss：

$$
L_Q = (Q_{\phi_1}(s,a)-y)^2 + (Q_{\phi_2}(s,a)-y)^2
$$

Actor loss 和 DDPG 相同：

$$
L_\mu=-\mathbb{E}[Q_{\phi_1}(s,\mu_\theta(s))]
$$

本仓库在 `src/drl_lab/algorithms/td3/losses.py` 中实现 min target：

```python
target_q1, target_q2 = target_critics(batch.next_obs, next_actions)
min_target_q = torch.minimum(target_q1, target_q2)
target = batch.rewards + gamma * (1.0 - batch.dones) * min_target_q
```

## 算法流程

1. 初始化 actor、twin critics 和对应 target networks。
2. 用 actor 加 exploration noise 与环境交互。
3. transition 存入 replay buffer。
4. 采样 batch。
5. 用 target actor 生成 next action，并加入 clipped noise。
6. 用 twin target critics 的较小值构造 TD target。
7. 更新两个 critics。
8. 每隔 `policy_delay` 次 critic update 才更新 actor。
9. Actor 更新时同步 soft update 所有 target networks。
10. 周期性 eval、checkpoint、导出 actor 和 twin critics。

## 代码双索引

- Demo：`curriculum/chapters/05_policy_gradient/td3/code/td3_core_demo.py`
- Twin critics：`src/drl_lab/algorithms/td3/networks.py`
- TD3 loss：`src/drl_lab/algorithms/td3/losses.py`
- Agent：`src/drl_lab/algorithms/td3/agent.py`
- Train：`src/drl_lab/algorithms/td3/train.py`
- DDPG actor/soft update 复用：`src/drl_lab/algorithms/ddpg`
- 测试：`tests/test_td3.py`

建议先比较 `ddpg/losses.py` 和 `td3/losses.py`，看 TD3 target 多了哪些步骤。

## 实验任务

```bash
conda run -n drl-lab python curriculum/chapters/05_policy_gradient/td3/code/td3_core_demo.py
conda run -n drl-lab python -m drl_lab.algorithms.td3.train --total-steps 300 --learning-starts 32 --eval-episodes 1
```

观察：

- 两个 critic 输出 shape 是否一致；
- target action noise 是否被裁剪；
- actor 是否延迟更新；
- critic loss 是否 finite；
- actor 和 twin critics 是否导出 ONNX。

## Debug Checklist

- twin critics 是否独立初始化，而不是共享同一模块。
- target 是否取 `min(q1, q2)`，不是 max 或平均。
- target action noise 是否 clamp 到 `[-noise_clip, noise_clip]`。
- noisy target action 是否再 clamp 到 action limit。
- actor update 是否按 `policy_delay` 触发。
- soft update 是否在 actor update 节奏下执行。
- twin critic ONNX 是否输出 `q1` 和 `q2`。

## Spinning Up 对照

Spinning Up TD3 是本章主参考。阅读时重点看三项 trick：clipped double Q-learning、delayed policy update、target policy smoothing。本仓库将这些 trick 拆成可测试函数，并补充：

- twin critic shape 和输出名；
- actor/twin critic export；
- 短训练 artifact；
- 与 DDPG 共享组件的代码对照。

## 学完标准

- 能说明 TD3 相对 DDPG 修了哪三个问题。
- 能写出 TD3 target。
- 能解释为什么取 twin critics 的最小值。
- 能在源码中指出 target smoothing 和 delayed update。
- 能跑通 TD3 demo 和短训练。
- 能完成 `curriculum/chapters/05_policy_gradient/td3/report.md`。
