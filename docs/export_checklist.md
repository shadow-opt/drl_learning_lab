# Export Checklist

Use this checklist for every trainable neural network.

- Call `model.eval()` before export.
- Use `torch.inference_mode()` for reference outputs.
- Save or document the example input shape.
- Record PyTorch, ONNX, and ONNXRuntime versions.
- Decide whether batch dimension is fixed or dynamic.
- Export only inference behavior, not the training loop.
- Compare PyTorch and ONNXRuntime outputs.
- Compare PyTorch and TorchScript outputs when using TorchScript.
- Compare PyTorch and `torch.export` outputs when saving an ExportedProgram.
- Record `max_abs_diff` and `mean_abs_diff`.
- Test actor outputs for policy networks.
- Test Q-values for value networks.
- Export learned value functions and critics, not only actors.
- For multi-input critics, name inputs explicitly, for example `obs` and
  `actions`.
- Keep export scripts deterministic and independent from training.
