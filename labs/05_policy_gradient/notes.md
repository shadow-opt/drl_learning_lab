# 05 Policy Gradient Lab Guide

## 前置阅读

先读 `course/05_policy_gradient/index.md`，再按顺序进入 VPG、PPO、TRPO、DDPG、TD3、SAC 子章节。

## 实验目标

- 理解 policy gradient、trust region、actor-critic 和 maximum entropy 的学习路线。
- 入门验收：跑通 VPG 和 PPO core demo。
- 完整验收：再跑通 DDPG、TD3、SAC core demo，并完成 TRPO math demo。
- 对照 `src/drl_lab/algorithms/*` 阅读工程实现。

## 代码入口

```bash
conda run -n drl-lab python labs/05_policy_gradient/code/list_algorithms.py
```

各算法 demo 在 `labs/05_policy_gradient/<algo>/code/`。

## 提交产物

- 完成总览 exercises。
- 入门阶段至少完成 VPG、PPO 两份子目录 report。
- 完整阶段再补 DDPG、TD3、SAC、TRPO report。
- 说明每个算法相对前一个算法解决了什么问题；TRPO 只要求数学和 PPO 关系，不要求完整 trainer。

## 常见坑

- 把 on-policy 和 off-policy 数据混用。
- advantage、log prob、action shape 对不上。
- 连续动作 actor 的 deterministic eval 和 stochastic training 混淆。
