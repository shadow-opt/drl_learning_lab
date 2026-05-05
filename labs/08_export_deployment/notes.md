# 08 Export Deployment Lab Guide

## 前置阅读

先读 `course/08_export_deployment.md` 和 `docs/export_checklist.md`。

## 实验目标

- 跑通 export demo。
- 理解 TorchScript、`torch.export`、ONNX 和 ONNXRuntime 的区别。
- 检查 PyTorch vs ONNXRuntime 一致性。

## 代码入口

```bash
conda run -n drl-lab python labs/08_export_deployment/code/export_demo.py
```

期望输出包含 `passed=True`、`max_abs_diff` 和 artifact 路径。`max_abs_diff=0` 或接近 0 说明这个小模型在当前输入上通过了一致性检查。

工程实现：

- `src/drl_lab/common/export.py`
- `src/drl_lab/common/onnx_check.py`

算法训练入口会把最终导出信息写入 run 目录的 `export_report.json`，最终评估写入 `eval_result.json`。

## 提交产物

- 完成 `exercises.md`。
- 填写 `report.md`。
- 选择一个算法模型说明其导出输入输出。

## 常见坑

- 导出训练路径而不是推理路径。
- stochastic policy 导出时仍然采样。
- 多输入 critic 没有命名输入。
- dynamic batch 维没有设置。
