# DDPG Spinning Up Mapping

Spinning Up role:

- introduces deterministic policy gradients for continuous control
- explains target networks and Polyak averaging
- uses replay-buffer off-policy actor-critic training

Local lab role:

- implement reusable continuous replay batches
- implement a bounded deterministic actor
- implement a continuous-action Q-network
- test critic loss, actor loss, Polyak update, and ONNX actor export

Expected differences:

- code is split into buffer, networks, losses, demos, and tests
- inference export is a first-class requirement
- the current lab slice is core components, not a full benchmark training loop
