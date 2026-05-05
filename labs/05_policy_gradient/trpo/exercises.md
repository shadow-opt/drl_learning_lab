# TRPO Exercises

1. Derive the quadratic KL approximation used in TRPO.
2. Use `conjugate_gradient` to solve a 2x2 positive definite system by hand and
   compare it with PyTorch.
3. Change `max_kl` in `scale_step_to_kl` and verify the quadratic KL changes as
   expected.
4. Explain how PPO clipping approximates some of the same caution as TRPO's
   trust region.
5. Optional: add a line-search demo for a tiny categorical policy.
