# Experiment Report

这是一份通用实验报告模板。每个 lab 的 `report.md` 可以更具体，但至少要覆盖这里的核心信息。

## 问题

这次实验想验证什么？

## 设置

- Algorithm：
- Environment：
- Seed：
- Hardware：
- Dependency versions：
- Key hyperparameters：

## 预期

运行前预计会看到什么曲线、分数或失败模式？

## 结果

记录 reward curve、loss curve、eval score、训练时间和主要 artifact。

## Debug 记录

- 现象：
- 初步假设：
- 定位过程：
- root cause：
- 修复方法：

## Export 检查

- ONNX file：
- ONNX input/output names：
- Example input shape：
- Output shape：
- PyTorch vs ONNXRuntime max abs diff：
- PyTorch vs ONNXRuntime mean abs diff：
- eval_result.json：
- export_report.json：

## 结论

这次实验学到了什么？下一次应该改什么？
