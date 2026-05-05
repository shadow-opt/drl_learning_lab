# SAC Report

## Objective

Implement and test the SAC core update equations and a minimal Pendulum
training loop.

## Experiment Setup

- Environment:
- Seeds:
- Actor hidden sizes:
- Critic hidden sizes:
- Alpha:
- Target entropy:

## Metrics

- Critic loss:
- Actor loss:
- Temperature loss:
- Alpha:
- Eval return:
- ONNX max absolute difference:

## Smoke Command

```bash
conda run -n drl-lab python labs/05_policy_gradient/sac/code/sac_core_demo.py
conda run -n drl-lab python -m drl_lab.algorithms.sac.train --total-steps 300 --learning-starts 32 --eval-episodes 1
```

## Findings

- 

## Debug Notes

- Check sampled actions stay inside action bounds.
- Check log probabilities are finite after the tanh correction.
- Check target Q shape is `[batch]`.
- Check `log_alpha` receives gradients while sampled log probabilities are
  detached in the temperature loss.
