# PyTorch Foundations Exercises

## 纸笔题

1. 解释 `model.train()` 和 `model.eval()` 的区别。
2. 解释 `torch.no_grad()`、`torch.inference_mode()` 和 `detach()` 的区别。
3. 写出本仓库 RL batch shape 约定：`obs`、`action`、`reward`、`done`。
4. 说明为什么导出时要使用 deterministic inference graph。

## 代码题

1. 跑通 `mlp_export_demo.py`。
2. 改变 example input 的 batch size，观察 ONNX 一致性是否仍通过。
3. 打开 `src/drl_lab/common/export.py`，找到 ONNX、TorchScript、`torch.export` 的函数入口。
4. 打开 `tests/test_onnx_consistency.py`，说明测试比较了什么。

## 观察题

1. PyTorch 与 ONNXRuntime 最大误差是多少？
2. 如果输入 dtype 改错，错误会在哪里暴露？
3. checkpoint 恢复后，哪些状态必须一致？

## 提交物

- 填完 `report.md`。
- 记录一次导出命令和一致性结果。
- 写下一个 shape 或 dtype 检查点。
