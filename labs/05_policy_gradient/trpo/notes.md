# TRPO Lab Guide

## 前置阅读

先读 `course/05_policy_gradient/trpo.md`。本 lab 是数学阅读模块，不是完整 benchmark trainer。

## 实验目标

- 跑通 TRPO math demo。
- 理解 conjugate gradient 和 KL step scaling。
- 对比 TRPO trust region 和 PPO clipping。

## 代码入口

```bash
conda run -n drl-lab python labs/05_policy_gradient/trpo/code/trpo_math_demo.py
```

工程实现：`src/drl_lab/algorithms/trpo/math.py`。测试：`tests/test_trpo.py`。

## 提交产物

- 写出 KL 约束。
- 填写 `report.md`。
- 说明为什么当前不实现完整 TRPO trainer。

## 常见坑

- 把 conjugate gradient 当成普通梯度下降。
- 忽略 KL 方向和 step scaling。
- 把 TRPO 和 PPO 当成无关算法。
