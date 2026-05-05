# TD3 Lab Guide

## 前置阅读

先读 `course/05_policy_gradient/td3.md`。TD3 应在 DDPG 之后学习。

## 实验目标

- 跑通 TD3 core demo。
- 检查 twin critics、target policy smoothing 和 delayed actor update。
- 跑一个短 Pendulum 训练。

## 代码入口

```bash
conda run -n drl-lab python labs/05_policy_gradient/td3/code/td3_core_demo.py
conda run -n drl-lab python -m drl_lab.algorithms.td3.train --total-steps 300 --learning-starts 32 --eval-episodes 1
```

工程实现：`src/drl_lab/algorithms/td3`。测试：`tests/test_td3.py`。

## 提交产物

- 写出 TD3 target。
- 填写 `report.md`。
- 解释 TD3 相对 DDPG 的三项修正。

## 常见坑

- target 没有取 twin critics 的最小值。
- actor 没有延迟更新。
- target action noise 没有裁剪。
