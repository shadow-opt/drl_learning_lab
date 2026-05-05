# TRPO Notes

TRPO is included here as a math-reading module rather than a full benchmark
implementation. It is useful because it explains the trust-region view behind
PPO.

## Core Ideas

- Optimize a policy objective while constraining the average KL divergence.
- Approximate the KL constraint with a local quadratic model.
- Use the natural-gradient direction instead of the raw policy gradient.
- Solve the natural-gradient system with conjugate gradient.
- Use a line search in full implementations to reject bad steps.

## Local Code

`src/drl_lab/algorithms/trpo/math.py` contains:

- `conjugate_gradient`
- `scale_step_to_kl`

These functions are intentionally small and isolated so the math can be tested
without adding a fragile full TRPO training loop.
