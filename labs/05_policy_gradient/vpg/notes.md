# VPG Notes

Vanilla Policy Gradient is the first policy-gradient algorithm in the Spinning
Up sequence.

## Required Implementation Pieces

- categorical actor for discrete actions
- value function baseline
- trajectory buffer
- reward-to-go
- advantage normalization
- actor loss
- value loss
- actor ONNX export

## Implemented

- categorical actor
- value function baseline
- discounted cumulative sums
- trajectory batch builder
- advantage normalization
- policy and value losses
- actor ONNX export consistency test
