# TD3 Spinning Up Mapping

Spinning Up role:

- explains DDPG failure modes from overestimated Q-values
- introduces clipped double Q-learning
- introduces target policy smoothing and delayed actor updates

Local lab role:

- reuse the DDPG replay buffer and deterministic actor
- add twin continuous Q-networks
- test clipped target actions and twin-critic Bellman loss
- export the deterministic actor path to ONNX

Expected differences:

- target smoothing is tested independently
- the twin critic is a reusable module for SAC
- the current lab slice is core components, not a full benchmark training loop
