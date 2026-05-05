# Iteration Guide

This lab should grow through small complete learning slices, not large empty
directory trees.

## Definition of Done for a Learning Slice

Each slice is complete only when it has:

- notes explaining the concept
- exercises with acceptance criteria
- runnable code
- at least one report or report template entry
- tests or a smoke command
- checkpoint/eval/export handling when a trainable neural network is involved
- git commit after tests pass

## Recommended Git Loop

```bash
git status
conda run -n drl-lab python -m pytest
conda run -n drl-lab ruff check .
conda run -n drl-lab mypy src
git add .
git commit -m "Complete <learning-slice>"
```

## Growth Policy

- Add DQN before policy gradients.
- Add VPG before PPO.
- Add PPO before SAC.
- Keep TRPO as a math-reading module until the rest of the lab is stable.
- Do not add robotics, Isaac Gym, sim2real, or distributed training.
