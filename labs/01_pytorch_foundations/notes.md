# PyTorch Foundations Notes

This module trains the PyTorch engineering habits needed for deep RL.

## Required Topics

- tensor creation and shape inspection
- broadcasting
- autograd
- `nn.Module`
- optimizers
- train/eval mode
- `torch.no_grad` vs `torch.inference_mode`
- device and dtype management
- checkpoint and resume
- ONNX export and ONNXRuntime consistency testing

## Shape Annotation Standard

Every model forward method should document expected shapes:

```text
input:  [batch, feature_dim]
output: [batch, output_dim]
```

For RL later:

```text
obs:     [batch, obs_dim]
action:  [batch, action_dim] or [batch]
reward:  [batch]
done:    [batch]
```
