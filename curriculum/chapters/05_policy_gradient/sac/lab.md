# SAC Lab Guide

## 前置阅读

先读 `curriculum/chapters/05_policy_gradient/sac/lesson.md`。SAC 应在 DDPG/TD3 后学习。

## 实验目标

- 跑通 SAC core demo。
- 检查 squashed Gaussian actor、twin critics、entropy term 和 alpha。
- 跑一个短 Pendulum 训练并检查 ONNX export。

## 代码入口

```bash
conda run -n drl-lab python curriculum/chapters/05_policy_gradient/sac/code/sac_core_demo.py
conda run -n drl-lab python -m drl_lab.algorithms.sac.train --total-steps 300 --learning-starts 32 --eval-episodes 1
```

工程实现：`src/drl_lab/algorithms/sac`。测试：`tests/test_sac.py`。

## 提交产物

- 写出 SAC critic target 和 actor objective。
- 填写 `report.md`。
- 说明 actor 训练采样路径和导出 deterministic 路径的区别。

## 常见坑

- tanh log prob correction 漏掉。
- alpha loss 符号或 detach 错误。
- 导出时仍包含 stochastic sampling。
