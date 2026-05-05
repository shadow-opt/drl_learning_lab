# Export and Deployment Notes

Every trainable model should eventually run outside the training loop.

## Required Topics

- `model.eval()`
- `torch.inference_mode()`
- TorchScript overview
- `torch.export`
- ONNX export
- ONNXRuntime inference
- numerical consistency thresholds
- fixed and dynamic batch dimensions

## Current Repository Examples

- MLP export demo
- linear regression export
- binary classifier export
- DQN Q-network export
