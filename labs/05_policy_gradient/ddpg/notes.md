# DDPG Notes

DDPG extends actor-critic methods to deterministic continuous-control policies.

## Required Implementation Pieces

- deterministic actor
- continuous-action Q-network
- continuous replay buffer
- target actor and target critic
- Bellman critic loss
- deterministic actor loss
- soft target updates
- actor ONNX export

## Implemented

- deterministic bounded actor
- continuous Q-network
- continuous replay buffer
- actor and critic losses
- soft target update
- actor ONNX export consistency test
