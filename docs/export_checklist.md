# Export Checklist

每个可训练神经网络最终都要能离开训练循环独立推理。导出前先确认以下事项。

## 基础检查

- 导出前调用 `model.eval()`。
- 用 `torch.inference_mode()` 生成 PyTorch reference output。
- 记录 example input shape。
- 记录 model output shape。
- 记录 input/output names。
- 记录 PyTorch、ONNX、ONNXRuntime 版本。
- 明确 batch 维是 fixed 还是 dynamic。
- 只导出 inference behavior，不导出 training loop。

## 一致性检查

- 比较 PyTorch 和 ONNXRuntime 输出。
- 使用 TorchScript 时，比较 PyTorch 和 TorchScript 输出。
- 使用 `torch.export` 时，比较 PyTorch 和 ExportedProgram 输出。
- 记录 `max_abs_diff` 和 `mean_abs_diff`。
- 将导出路径和误差写入 `export_report.json` 或实验 report。

## RL 特有检查

- policy network 导出 actor output。
- value network 导出 value 或 Q-value。
- 不只导出 actor；需要时也导出 value function 和 critic。
- stochastic policy 导出时使用 deterministic inference path。
- multi-input critic 必须显式命名输入，例如 `obs` 和 `actions`。
- export demo 必须 deterministic，并且不依赖训练中的环境状态。
