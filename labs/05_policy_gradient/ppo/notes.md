# PPO Notes

PPO extends policy gradients with a clipped surrogate objective.

## Required Implementation Pieces

- actor and critic networks
- trajectory buffer
- GAE-lambda
- log-prob ratio
- clipped policy objective
- value loss
- entropy bonus
- KL monitoring
- actor ONNX export

## Implemented

- GAE-lambda
- PPO trajectory batch
- clipped policy objective
- log-prob ratio diagnostics
- approximate KL diagnostic
- entropy diagnostic
- actor ONNX export consistency test
