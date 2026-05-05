# drl_learning_lab

`drl_learning_lab` is a long-term self-study lab for deep reinforcement
learning. It is intentionally designed as both a study guide and an engineering
training repository.

The first goal is not robotics, Isaac Gym, or large-scale simulation. The first
goal is to build the foundations:

- machine learning and PyTorch fundamentals
- reinforcement learning math
- tabular RL
- modern PyTorch deep RL implementations
- experiment management and debugging
- TorchScript, `torch.export`, ONNX, and ONNXRuntime export validation

OpenAI Spinning Up is used as the main deep RL reference, but this repository is
not a copy of Spinning Up. The goal is to absorb, index, rewrite, and extend the
material with modern PyTorch engineering practice.

## Environment

Conda is the recommended environment manager.

```bash
conda env create -f environment.yml
conda activate drl-lab
python -m pip install -e ".[dev]"
pytest
```

If `python` is not available outside the environment, use:

```bash
conda run -n drl-lab python -m pytest
```

## Git Workflow

Use git from the beginning:

```bash
git status
git add .
git commit -m "Initialize DRL learning lab"
```

Recommended iteration loop:

1. Create or update a small learning module.
2. Add notes, exercises, code, and a short report.
3. Run tests.
4. Commit the completed learning slice.

## Repository Shape

```text
docs/                 Roadmap, checklists, templates, resources
external/spinningup/  Source and license notes for Spinning Up
src/drl_lab/          Reusable engineering code
labs/                 Learning modules with notes, exercises, code, report
experiments/          Configs, runs, and reports
tests/                Unit and smoke tests
```

See [docs/roadmap.md](docs/roadmap.md) for the full learning path.
See [docs/first_month_plan.md](docs/first_month_plan.md) for the first
executable month.
See [docs/iteration_guide.md](docs/iteration_guide.md) for the definition of
done and recommended git loop.

## First Executable Slice

The current skeleton includes:

- seed, device, checkpoint, export, and ONNX consistency helpers
- a small MLP example
- linear regression, binary classification, and image classification training
  examples
- GridWorld value iteration, policy iteration, Monte Carlo control, SARSA,
  Q-learning, and Expected SARSA
- DQN components: replay buffer, Q-network, DQN loss, ONNX export test
- DQN CartPole training loop with checkpoint, eval, and Q-network export
- VPG and PPO CartPole training loops
- DDPG Pendulum training loop
- TD3 Pendulum training loop
- SAC Pendulum training loop
- TRPO math-reading utilities for conjugate gradient and KL step scaling
- TorchScript, `torch.export`, ONNX, and ONNXRuntime consistency checks
- pytest coverage for deterministic seeding, checkpoint restore, ONNX export,
  supervised examples, tabular RL, DQN, VPG, PPO, DDPG, TD3, SAC, and TRPO

This remains intentionally incremental. Future algorithm modules should be added
only when their notes, exercises, code, report, and tests are ready.

## Current Quality Gates

```bash
conda run -n drl-lab python -m pytest
conda run -n drl-lab ruff check .
conda run -n drl-lab mypy src
```

Useful demos:

```bash
conda run -n drl-lab python labs/00_ml_foundations/code/linear_regression.py
conda run -n drl-lab python labs/00_ml_foundations/code/binary_classifier.py
conda run -n drl-lab python labs/00_ml_foundations/code/image_classifier.py
conda run -n drl-lab python labs/02_rl_math/code/value_iteration_demo.py
conda run -n drl-lab python labs/03_tabular_rl/code/td_control_demo.py
conda run -n drl-lab python labs/04_dqn/code/dqn_smoke_demo.py
conda run -n drl-lab python -m drl_lab.algorithms.dqn.train --total-steps 300 --learning-starts 32
conda run -n drl-lab python labs/05_policy_gradient/code/list_algorithms.py
conda run -n drl-lab python labs/05_policy_gradient/vpg/code/vpg_core_demo.py
conda run -n drl-lab python labs/05_policy_gradient/ppo/code/ppo_core_demo.py
conda run -n drl-lab python labs/05_policy_gradient/ddpg/code/ddpg_core_demo.py
conda run -n drl-lab python -m drl_lab.algorithms.ddpg.train --total-steps 300 --learning-starts 32 --eval-episodes 1
conda run -n drl-lab python labs/05_policy_gradient/td3/code/td3_core_demo.py
conda run -n drl-lab python -m drl_lab.algorithms.td3.train --total-steps 300 --learning-starts 32 --eval-episodes 1
conda run -n drl-lab python labs/05_policy_gradient/sac/code/sac_core_demo.py
conda run -n drl-lab python -m drl_lab.algorithms.sac.train --total-steps 300 --learning-starts 32 --eval-episodes 1
conda run -n drl-lab python labs/05_policy_gradient/trpo/code/trpo_math_demo.py
conda run -n drl-lab python labs/06_spinningup_track/code/list_mappings.py
conda run -n drl-lab python labs/07_experiment_engineering/code/snapshot_demo.py
conda run -n drl-lab python labs/08_export_deployment/code/export_demo.py
```
