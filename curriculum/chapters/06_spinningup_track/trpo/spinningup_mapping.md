# TRPO Spinning Up Mapping

Spinning Up role:

- introduces trust-region policy optimization
- motivates KL-constrained policy updates
- explains conjugate gradient and line search in the policy-gradient setting

Local lab role:

- treat TRPO as a math-reading module
- implement small, tested utilities for conjugate gradient and KL step scaling
- use the TRPO reading to understand why PPO is a simpler practical algorithm

Expected differences:

- no full TRPO trainer is included in the first lab version
- code focuses on the reusable math pieces rather than reproducing Spinning Up
  source structure
