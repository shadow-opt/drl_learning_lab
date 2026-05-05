# PPO Report

PPO tensors, GAE, clipped objective, diagnostics, CartPole training,
checkpointing, eval, and actor ONNX export are implemented.

## Smoke Command

```bash
conda run -n drl-lab python labs/05_policy_gradient/ppo/code/ppo_core_demo.py
conda run -n drl-lab python -m drl_lab.algorithms.ppo.train --epochs 1 --steps-per-epoch 128
```
