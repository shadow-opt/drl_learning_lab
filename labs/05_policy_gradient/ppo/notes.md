# PPO Lab Guide

## 前置阅读

先读 `course/05_policy_gradient/ppo.md`。PPO 应在 VPG 之后学习。

## 实验目标

- 跑通 PPO core demo。
- 检查 old log prob、ratio、clip fraction、approx KL。
- 理解 clipped objective 如何限制 policy update。

## 代码入口

```bash
conda run -n drl-lab python labs/05_policy_gradient/ppo/code/ppo_core_demo.py
```

工程实现：`src/drl_lab/algorithms/ppo`。测试：`tests/test_ppo.py`。

## 提交产物

- 写出 PPO clipped objective。
- 填写 `report.md`。
- 解释 PPO 和 TRPO 的关系。

## 常见坑

- old log prob 没有在采样时保存。
- ratio 直接相除概率导致数值不稳定。
- advantage 没有标准化。
