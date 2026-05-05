# TRPO Report

TRPO is tracked as a math-reading module. The repository includes tested
conjugate-gradient and KL-step-scaling utilities, but it does not include a full
TRPO trainer.

## Smoke Command

```bash
conda run -n drl-lab python labs/05_policy_gradient/trpo/code/trpo_math_demo.py
conda run -n drl-lab python -m pytest tests/test_trpo.py
```
