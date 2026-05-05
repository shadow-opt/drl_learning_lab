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
- multi-input critic export with named inputs

## Current Repository Examples

- MLP export demo
- linear regression export
- binary classifier export
- DQN Q-network export
- VPG policy and value-function export
- PPO policy and value-function export
- DDPG actor and critic export
- TD3 actor and twin-critic export
- SAC actor and twin-critic export
- TorchScript trace export
- `torch.export` ExportedProgram export
- ONNXRuntime consistency checks
- TorchScript consistency checks
- `torch.export` consistency checks
