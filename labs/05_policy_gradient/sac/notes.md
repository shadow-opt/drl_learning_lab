# SAC Notes

Soft Actor-Critic is an off-policy actor-critic algorithm for continuous action
spaces. It extends clipped double Q-learning with a stochastic actor and an
entropy bonus, so the policy is rewarded for both high value and sufficient
exploration.

## Core Equations

Critic target:

```text
y = r + gamma * (1 - done) * (min(Q1_target(s', a'), Q2_target(s', a')) - alpha * log pi(a'|s'))
```

Actor objective:

```text
J_pi = E[alpha * log pi(a|s) - min(Q1(s, a), Q2(s, a))]
```

Temperature objective:

```text
J_alpha = -log_alpha * stop_gradient(log pi(a|s) + target_entropy)
```

## Implementation Notes

- The actor is a reparameterized Gaussian followed by `tanh`.
- The log probability must include the `tanh` correction.
- `forward` returns deterministic mean actions and is the export path.
- `sample` is used only during training losses.
- Replay, target critics, and Polyak updates reuse the DDPG/TD3 components.

## Spinning Up Mapping

Spinning Up's SAC chapter is the conceptual reference. This lab keeps the same
algorithmic ideas but uses modern PyTorch modules, explicit shape checks, small
unit tests, and ONNX export for the deterministic inference actor.
