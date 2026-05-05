# Roadmap

This roadmap keeps the lab small at the start and grows it only when a learning
slice has notes, exercises, code, reports, and tests.

## Phase 0: Engineering Baseline

Goal: make the repository runnable, testable, and reproducible.

Deliverables:

- conda environment
- editable Python package under `src/drl_lab`
- seed, device, checkpoint, export, and ONNX consistency helpers
- pytest smoke tests
- git workflow documented in `README.md`

Exercises:

- Set a seed and prove two model initializations match.
- Save and reload a checkpoint.
- Export a small MLP to ONNX and compare PyTorch vs ONNXRuntime outputs.

## Phase 1: ML and PyTorch Foundations

Goal: build enough supervised learning and PyTorch engineering skill to make
deep RL implementation manageable.

Topics:

- tensor shape annotation and broadcasting
- autograd, optimizers, and schedulers
- train/eval mode
- `torch.no_grad` and `torch.inference_mode`
- `Dataset` and `DataLoader`
- device and dtype management
- checkpoint, resume, eval, and export

Deliverables:

- linear regression
- binary classifier MLP
- MNIST or FashionMNIST classifier
- ONNX export and consistency tests for each trainable model

Current status:

- linear regression is implemented
- binary classifier is implemented
- a small 28x28 image classifier is implemented with an offline synthetic
  dataset; swapping in MNIST or FashionMNIST remains an optional extension when
  `torchvision` is available

## Phase 2: RL Math

Goal: understand the math before adding neural networks.

Topics:

- MDPs
- return and discounting
- value functions
- Bellman expectation and optimality equations
- policy evaluation and improvement
- Monte Carlo vs temporal difference learning
- bias and variance
- advantage and GAE

Deliverables:

- handwritten notes
- small GridWorld examples
- value iteration and policy iteration code

Current status:

- deterministic GridWorld is implemented
- value iteration is implemented
- policy iteration is implemented

## Phase 3: Tabular RL

Goal: learn the update rules without neural network complexity.

Algorithms:

- Monte Carlo control
- SARSA
- Q-learning
- Expected SARSA

Deliverables:

- FrozenLake or CliffWalking experiments
- epsilon-greedy comparisons
- convergence reports

Current status:

- local GridWorld value iteration, policy iteration, Monte Carlo control,
  Q-learning, SARSA, and Expected SARSA are implemented

## Phase 4: DQN

Goal: bridge PyTorch engineering and RL.

Required components:

- replay buffer
- target network
- epsilon schedule
- Huber loss
- gradient clipping
- checkpoint and eval
- Q-network ONNX export

Current status:

- replay buffer, Q-network, DQN loss, smoke demo, and ONNX consistency tests are
  implemented
- CartPole training loop, checkpoint, eval hook, and Q-network export are
  implemented

## Phase 5: Spinning Up Main Track

Goal: study Spinning Up as the DRL conceptual backbone while writing modern
PyTorch implementations.

Order:

1. VPG
2. PPO
3. DDPG
4. TD3
5. SAC
6. TRPO as math reading and optional implementation

Each algorithm needs:

- notes
- exercises
- implementation
- experiment report
- Spinning Up mapping note
- ONNX export test for inference networks

Current status:

- VPG core components and CartPole training loop are implemented
- PPO core components and CartPole training loop are implemented
- DDPG core components and Pendulum training loop are implemented
- TD3 core components are implemented
- SAC core components are implemented

## Phase 6: Experiment Engineering

Goal: make experiments reproducible and debuggable.

Topics:

- config snapshots
- CSV or JSONL metrics
- TensorBoard
- seed discipline
- checkpoint structure
- independent eval scripts
- failure reports

## Phase 7: Export and Deployment

Goal: every trained model can run outside the training loop.

Topics:

- TorchScript overview
- `torch.export`
- ONNX export
- ONNXRuntime inference
- numerical consistency tests
- fixed vs dynamic batch shapes
