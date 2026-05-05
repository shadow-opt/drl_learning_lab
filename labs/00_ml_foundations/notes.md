# 00 ML Foundations Lab Guide

## 前置阅读

先读 `course/00_ml_foundations.md`。本 lab 不再重复完整教材，只负责把监督学习基础落到可运行实验。

## 实验目标

- 跑通线性回归、二分类和 28x28 图像分类 demo。
- 检查输入输出 shape、loss 曲线、checkpoint 和 ONNX 导出。
- 记录一次正常训练和一次故意制造的失败。

## 代码入口

```bash
conda run -n drl-lab python labs/00_ml_foundations/code/linear_regression.py
conda run -n drl-lab python labs/00_ml_foundations/code/binary_classifier.py
conda run -n drl-lab python labs/00_ml_foundations/code/image_classifier.py
```

工程工具在 `src/drl_lab/common/checkpoint.py`、`src/drl_lab/common/export.py` 和 `src/drl_lab/common/onnx_check.py`。

## 提交产物

- 完成 `exercises.md` 中的纸笔题、代码题和观察题。
- 填写 `report.md`。
- 确认 `tests/test_supervised_examples.py` 通过。

## 常见坑

- label dtype 和 loss function 不匹配。
- 忘记 `optimizer.zero_grad()`。
- eval 时没有 `model.eval()`。
- ONNX 输出没有和 PyTorch 输出比较。
