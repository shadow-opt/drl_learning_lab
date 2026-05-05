# Policy Gradient Track

This track starts the Spinning Up main line after DQN and tabular RL.

## Learning Order

1. VPG
2. PPO
3. DDPG
4. TD3
5. SAC
6. TRPO as optional math implementation

## Shared Concepts

- trajectory collection
- reward-to-go
- baseline and advantage
- policy loss
- value loss
- entropy
- GAE
- clipped objectives for PPO
- deterministic policy gradients
- target networks for off-policy actor-critic methods
- clipped double Q-learning and target policy smoothing

## Engineering Requirements

Every trainable actor or critic needs:

- shape annotations
- seed control
- train/eval separation
- checkpoint
- independent eval
- ONNX export for inference networks
- PyTorch vs ONNXRuntime consistency test
