# DQN Report

Use `docs/experiment_report_template.md` for DQN experiments.

## Current Status

The repository contains reusable DQN components, a CartPole training loop,
checkpointing, eval, and Q-network ONNX export consistency tests.

## Smoke Command

```bash
conda run -n drl-lab python -m drl_lab.algorithms.dqn.train --total-steps 300 --learning-starts 32
```
