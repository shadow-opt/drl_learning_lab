# DDPG Lab Guide

## 前置阅读

先读 `curriculum/chapters/05_policy_gradient/ddpg/lesson.md`。DDPG 是连续动作 actor-critic 的起点。

## 实验目标

- 跑通 DDPG core demo。
- 跑一个短 Pendulum 训练。
- 检查 actor、critic、target networks、replay buffer 和 ONNX export。

## 代码入口

```bash
conda run -n drl-lab python curriculum/chapters/05_policy_gradient/ddpg/code/ddpg_core_demo.py
conda run -n drl-lab python -m drl_lab.algorithms.ddpg.train --total-steps 300 --learning-starts 32 --eval-episodes 1
```

工程实现：`src/drl_lab/algorithms/ddpg`。测试：`tests/test_ddpg.py`。

## 提交产物

- 写出 critic target 和 actor objective。
- 填写 `report.md`。
- 说明 actor 和 critic ONNX 输入输出。

## 常见坑

- action 没有缩放到环境范围。
- target 计算没有 `no_grad`。
- actor loss 符号写错。
