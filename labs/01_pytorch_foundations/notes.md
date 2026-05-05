# 01 PyTorch Foundations Lab Guide

## 前置阅读

先读 `course/01_pytorch_foundations.md`。本 lab 聚焦 PyTorch 工程动作：shape、device、dtype、checkpoint、TorchScript、`torch.export`、ONNX。

## 实验目标

- 跑通 MLP 导出 demo。
- 理解同一个模型如何经过 PyTorch、TorchScript、`torch.export` 和 ONNXRuntime 验证。
- 修改 batch size，确认动态 batch 维是否正常。

## 代码入口

```bash
conda run -n drl-lab python labs/01_pytorch_foundations/code/mlp_export_demo.py
```

工程实现：

- `src/drl_lab/common/networks.py`
- `src/drl_lab/common/checkpoint.py`
- `src/drl_lab/common/export.py`
- `src/drl_lab/common/onnx_check.py`

## 提交产物

- 完成 `exercises.md`。
- 填写 `report.md`。
- 确认 export/checkpoint 相关测试通过。

## 常见坑

- 输入和模型不在同一 device。
- 导出路径包含随机采样或训练专用逻辑。
- 忘记比较导出模型和 PyTorch 原模型输出。
