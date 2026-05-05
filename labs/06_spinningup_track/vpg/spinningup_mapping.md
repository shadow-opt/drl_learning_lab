# VPG Spinning Up Mapping

Spinning Up role:

- introduces policy gradients
- explains reward-to-go and baselines
- provides the first policy-gradient implementation pattern

Local lab role:

- implement VPG with modern PyTorch modules
- add tests for returns and advantage normalization
- export the actor network to ONNX

Expected differences:

- use `src/` package structure
- keep CLI, agent, buffer, losses, and export concerns separate
- add PyTorch vs ONNXRuntime consistency tests
