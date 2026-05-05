# Spinning Up Track Report

Use this report to summarize what was learned from mapping Spinning Up into the
local lab structure.

## Current Status

- DQN is tracked as a local addition outside Spinning Up's classic algorithm
  set.
- VPG, PPO, DDPG, TD3, and SAC have local implementations and mapping notes.
- TRPO is represented as a math-reading module with tested utilities.

## Smoke Command

```bash
conda run -n drl-lab python labs/06_spinningup_track/code/list_mappings.py
```
