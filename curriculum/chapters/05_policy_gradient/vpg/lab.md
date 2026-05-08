# VPG Lab Guide

## 前置阅读

先读 `curriculum/chapters/05_policy_gradient/vpg/lesson.md`，再对照 Spinning Up VPG。

## 实验目标

- 跑通 VPG core demo。
- 检查 trajectory buffer、return-to-go、advantage normalization。
- 理解 policy loss 和 value loss 的输入 shape。

## 代码入口

```bash
conda run -n drl-lab python curriculum/chapters/05_policy_gradient/vpg/code/vpg_core_demo.py
```

工程实现：`src/drl_lab/algorithms/vpg`。测试：`tests/test_vpg.py`。

## 提交产物

- 写出 VPG policy loss。
- 填写 `report.md`。
- 说明 actor ONNX export 为什么只使用 deterministic inference path。

## 常见坑

- advantage 没有标准化。
- log prob 不是采样 action 的 log prob。
- policy loss 符号写反。
