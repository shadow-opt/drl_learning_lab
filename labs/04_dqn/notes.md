# DQN Notes

DQN is the first deep RL module because it connects PyTorch engineering with
tabular Q-learning.

## Required Components

- replay buffer
- Q-network
- target Q-network
- epsilon-greedy exploration
- Huber loss
- gradient clipping
- checkpoint, eval, and ONNX export

## Shape Standard

```text
obs:      [batch, obs_dim]
actions:  [batch]
rewards:  [batch]
next_obs: [batch, obs_dim]
dones:    [batch]
q_values: [batch, n_actions]
```
