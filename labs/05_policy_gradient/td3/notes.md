# TD3 Notes

TD3 improves DDPG with clipped double Q-learning, delayed policy updates, and
target policy smoothing.

## Required Implementation Pieces

- deterministic actor
- twin continuous Q-networks
- min-Q target
- target policy smoothing
- delayed actor update schedule
- soft target updates
- actor ONNX export

## Implemented

- twin continuous Q-network
- clipped noisy target actions
- TD3 twin critic loss
- actor loss reused from DDPG
- soft target update reused from DDPG
- actor ONNX export consistency test
