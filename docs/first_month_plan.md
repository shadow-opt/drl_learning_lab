# First Month Plan

This plan is the minimum executable path through the repository. Keep each week
small: read notes, run code, complete exercises, write a report, run quality
gates, then commit.

## Week 1: Engineering And ML Basics

- Run `conda env create -f environment.yml`.
- Run the quality gates from `README.md`.
- Study `labs/00_ml_foundations/notes.md`.
- Run linear regression, binary classification, and image classification demos.
- Confirm each run writes `config.json`, `environment.json`, `metrics.csv`,
  checkpoint, and ONNX files.

## Week 2: PyTorch Export And RL Math

- Run `labs/01_pytorch_foundations/code/mlp_export_demo.py`.
- Study `docs/export_checklist.md`.
- Run `labs/02_rl_math/code/value_iteration_demo.py`.
- Explain Bellman optimality and policy improvement in `labs/02_rl_math/report.md`.

## Week 3: Tabular RL And DQN

- Run all demos under `labs/03_tabular_rl/code/`.
- Compare Monte Carlo control, SARSA, Expected SARSA, and Q-learning.
- Run `labs/04_dqn/code/dqn_smoke_demo.py`.
- Run a short DQN CartPole training command from `README.md`.

## Week 4: Spinning Up Main Track

- Read `labs/06_spinningup_track/notes.md`.
- Run VPG and PPO core demos, then short training commands if needed.
- Run DDPG, TD3, and SAC core demos plus short Pendulum commands.
- Run the TRPO math demo and explain the KL-scaled step.
- Finish with `conda run -n drl-lab python -m pytest`,
  `conda run -n drl-lab ruff check .`, and `conda run -n drl-lab mypy src`.
