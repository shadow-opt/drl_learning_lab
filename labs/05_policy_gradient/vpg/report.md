# VPG Report

VPG tensors, losses, networks, CartPole training, checkpointing, eval, and
actor ONNX export are implemented.

## Smoke Command

```bash
conda run -n drl-lab python labs/05_policy_gradient/vpg/code/vpg_core_demo.py
conda run -n drl-lab python -m drl_lab.algorithms.vpg.train --epochs 1 --steps-per-epoch 128
```
