# SAC Spinning Up Mapping

Spinning Up role:

- explains maximum-entropy reinforcement learning
- introduces stochastic squashed Gaussian policies
- combines clipped double Q-learning with an entropy-regularized actor update

Local lab role:

- implement a tanh-squashed Gaussian actor with corrected log probabilities
- reuse the TD3 twin critic for minimum-Q targets
- test SAC critic, actor, and temperature losses
- export deterministic mean actions to ONNX for inference

Expected differences:

- stochastic sampling is kept out of the export path
- temperature learning is exposed as a separate tested loss
- the current lab slice is core components, not a full benchmark training loop
