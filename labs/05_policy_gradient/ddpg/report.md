# DDPG Report

DDPG core components and a minimal Pendulum training loop are implemented.

## Smoke Command

```bash
conda run -n drl-lab python labs/05_policy_gradient/ddpg/code/ddpg_core_demo.py
conda run -n drl-lab python -m drl_lab.algorithms.ddpg.train --total-steps 300 --learning-starts 32 --eval-episodes 1
```
