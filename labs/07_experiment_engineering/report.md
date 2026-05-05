# Experiment Engineering Report

Use this module to record improvements to logging, config, checkpointing, and
debugging.

## Current Status

- `drl_lab.common.experiment.save_run_snapshots` writes `config.json` and
  `environment.json`.
- Supervised learning, DQN, VPG, PPO, DDPG, TD3, and SAC training entrypoints
  save run snapshots before training.

## Smoke Command

```bash
conda run -n drl-lab python labs/07_experiment_engineering/code/snapshot_demo.py
```
