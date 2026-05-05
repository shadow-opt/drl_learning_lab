# Export Deployment Exercises

## 纸笔题

1. 解释 TorchScript、`torch.export`、ONNX 的区别。
2. 写出一致性测试的最大误差判断。
3. 说明为什么 policy 导出通常使用 deterministic action。

## 代码题

1. 跑通 export demo。
2. 打开 `src/drl_lab/common/export.py`，找到 multi-input ONNX export。
3. 打开 `tests/test_onnx_consistency.py`，说明 actor 和 critic 分别如何比较。

## 观察题

1. 导出模型的输入名和输出名是什么？
2. PyTorch 与 ONNXRuntime 最大误差是多少？
3. 动态 batch 是否可用？

## 提交物

- 填写 `report.md`。
- 记录一次导出一致性结果。
