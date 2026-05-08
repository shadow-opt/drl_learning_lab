# 05 Policy Gradient Track

这一轨道学习 Spinning Up 主线算法：VPG、PPO、DDPG、TD3、SAC，以及作为数学阅读模块的 TRPO。它们共同回答一个问题：除了学习 `Q(s,a)`，能不能直接学习 policy？

## 学习顺序

1. [VPG](vpg.md)：理解 policy gradient 的基本形式。
2. [PPO](ppo.md)：理解 clipped objective 和稳定 policy update。
3. [TRPO](trpo.md)：理解 KL 约束和 trust region 思想。
4. [DDPG](ddpg.md)：连续动作下的 deterministic actor-critic。
5. [TD3](td3.md)：修正 DDPG 的过估计和不稳定。
6. [SAC](sac.md)：最大熵 actor-critic。

## 共同学习闭环

- 先读算法讲义，写出核心 objective。
- 跑 `curriculum/chapters/05_policy_gradient/<algo>/code/` 的 demo。
- 对照 `src/drl_lab/algorithms/<algo>` 阅读工程实现。
- 运行短训练命令或核心测试。
- 在 report 中记录 shape、loss、return、导出结果。

## Spinning Up 对照

Spinning Up 是这一轨道的主要外部参考。每个算法页都会内嵌对应阅读方式，并指出本仓库相对 Spinning Up 额外补充的现代 PyTorch、测试、checkpoint 和 ONNX 导出能力。
