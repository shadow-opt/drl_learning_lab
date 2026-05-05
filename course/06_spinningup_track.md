# 06 Spinning Up 对照学习

## 学习目标

本章说明如何把 OpenAI Spinning Up 放进本仓库的学习系统。学完后你应该能为每个主线算法建立一条路径：Spinning Up 原文 -> 本仓库中文讲义 -> lab demo -> `src` 工程实现 -> tests/export -> report。

你还要明确一条边界：本仓库不是 Spinning Up 的 vendor copy。它使用 Spinning Up 做概念主线，同时补齐 ML/PyTorch 基础、DQN、实验工程和现代导出部署。

## 为什么重要

Spinning Up 是很好的 DRL 入门材料，但它不是完整自学仓库。它不会替你训练 PyTorch shape/device/dtype 习惯，也不系统覆盖 checkpoint、ONNX、ONNXRuntime、一致性测试和现代 `torch.export`。

如果只读 Spinning Up，你可能懂算法图解但写不出稳健工程实现；如果只看本仓库代码，你可能能跑脚本但不知道算法为什么这么写。对照学习就是把两者连起来。

## 核心直觉

把 Spinning Up 当成“外部概念老师”，把 `course/` 当成“中文重写和扩充”，把 `labs/` 当成“作业”，把 `src/` 当成“工程参考”。每学一个算法，都要完成四个动作：

1. 读懂外部解释。
2. 用中文讲义重建概念。
3. 跑本地 demo 和训练。
4. 读源码并写复盘。

这样做可以避免两个极端：只搬运外部材料，或者只堆本地脚本。

## 数学与公式

Spinning Up 主线反复使用这几个对象：

$$
\pi_\theta(a|s),\quad V^\pi(s),\quad Q^\pi(s,a),\quad A^\pi(s,a)
$$

阅读每章时都要问：

- 这个算法学习 policy 还是 value，或者两者都学？
- 数据是 on-policy 还是 off-policy？
- target 是 Monte Carlo return、TD target，还是 surrogate objective？
- 是否限制 policy update，比如 KL 或 clipping？
- 是否加入 entropy？

这些问题比背算法名更重要。

## 算法流程

每个 Spinning Up 对照学习单元按以下流程完成：

1. 读对应 `course` 算法章。
2. 读 Spinning Up 原文的算法说明。
3. 打开 `labs/06_spinningup_track/<algo>/spinningup_mapping.md`。
4. 跑对应算法 demo。
5. 阅读 `src/drl_lab/algorithms/<algo>`。
6. 跑对应测试或短训练。
7. 在 report 中写出概念和工程差异。

## 代码双索引

- Mapping 列表：`labs/06_spinningup_track/code/list_mappings.py`
- Mapping 文件：`labs/06_spinningup_track/*/spinningup_mapping.md`
- Spinning Up 来源：`external/spinningup`
- VPG/PPO/TRPO/DDPG/TD3/SAC：`course/05_policy_gradient`
- 工程实现：`src/drl_lab/algorithms`
- 测试：`tests/test_vpg.py`、`tests/test_ppo.py`、`tests/test_trpo.py`、`tests/test_ddpg.py`、`tests/test_td3.py`、`tests/test_sac.py`

推荐对照表：

```text
VPG  -> course/05_policy_gradient/vpg.md  -> src/drl_lab/algorithms/vpg
PPO  -> course/05_policy_gradient/ppo.md  -> src/drl_lab/algorithms/ppo
TRPO -> course/05_policy_gradient/trpo.md -> src/drl_lab/algorithms/trpo
DDPG -> course/05_policy_gradient/ddpg.md -> src/drl_lab/algorithms/ddpg
TD3  -> course/05_policy_gradient/td3.md  -> src/drl_lab/algorithms/td3
SAC  -> course/05_policy_gradient/sac.md  -> src/drl_lab/algorithms/sac
```

## 实验任务

```bash
conda run -n drl-lab python labs/06_spinningup_track/code/list_mappings.py
```

选择一个算法，例如 PPO，完成：

- 读 `course/05_policy_gradient/ppo.md`；
- 读 Spinning Up PPO；
- 打开 `labs/06_spinningup_track/ppo/spinningup_mapping.md`；
- 跑 `labs/05_policy_gradient/ppo/code/ppo_core_demo.py`；
- 在 `labs/06_spinningup_track/report.md` 写出差异。

## Debug Checklist

- 是否保留上游来源和 MIT license。
- 是否复制了大段 Spinning Up 源码。
- 是否能指出本仓库对应的 course、lab、src、tests。
- 是否把 DQN 误当成 Spinning Up 主线内容。
- 是否把旧实现风格当成必须遵守的现代 PyTorch 风格。
- 是否在 report 中记录了本仓库额外工程能力。

## Spinning Up 对照

本章本身就是 Spinning Up 对照层。具体算法的对照说明内嵌在各算法章节中：

- VPG：policy gradient、reward-to-go、baseline。
- PPO：clipped objective、GAE、KL early stopping。
- TRPO：trust region、conjugate gradient、line search。
- DDPG：deterministic actor-critic、target networks。
- TD3：clipped double Q、delayed update、target smoothing。
- SAC：maximum entropy、squashed Gaussian actor、temperature。

DQN 是本仓库额外补充，不来自 Spinning Up 主线。

## 学完标准

- 能解释 Spinning Up 在本仓库中的角色。
- 能为任一算法列出“外部阅读 -> 中文讲义 -> lab demo -> src 实现 -> tests/export -> report”的路径。
- 能说明本仓库相对 Spinning Up 补了哪些基础和工程能力。
- 能遵守 license/source discipline。
