# First Month Plan

这个计划不是限制学习范围，而是给第一次穿过仓库的最小路径。完整路线见 `course/index.md`。

## Week 1：入口、环境、ML 基础

目标：知道仓库怎么学，并跑通最小监督学习闭环。

任务：

1. 阅读 `README.md` 和 `course/index.md`。
2. 创建环境并运行质量门禁。
3. 阅读 `course/00_ml_foundations.md`。
4. 完成 `labs/00_ml_foundations` 的三个 demo。
5. 填写 `labs/00_ml_foundations/report.md`。

命令：

```bash
conda env create -f environment.yml
conda run -n drl-lab python -m pytest
conda run -n drl-lab python labs/00_ml_foundations/code/linear_regression.py
```

## Week 2：PyTorch 工程与 RL 数学

目标：理解 shape、checkpoint、export，以及 Bellman backup。

任务：

1. 阅读 `course/01_pytorch_foundations.md`。
2. 跑通 MLP export demo。
3. 阅读 `course/02_rl_math.md`。
4. 跑 value iteration demo。
5. 在 report 中手算一个 Bellman backup。

命令：

```bash
conda run -n drl-lab python labs/01_pytorch_foundations/code/mlp_export_demo.py
conda run -n drl-lab python labs/02_rl_math/code/value_iteration_demo.py
```

## Week 3：Tabular RL 与 DQN

目标：从表格 update rule 过渡到 DQN。

任务：

1. 阅读 `course/03_tabular_rl.md`。
2. 跑通所有 tabular demos。
3. 阅读 `course/04_dqn.md`。
4. 跑 DQN smoke demo 和短 CartPole 训练。
5. 比较 tabular Q-learning target 与 DQN target。

命令：

```bash
conda run -n drl-lab python labs/03_tabular_rl/code/q_learning_demo.py
conda run -n drl-lab python labs/04_dqn/code/dqn_smoke_demo.py
conda run -n drl-lab python -m drl_lab.algorithms.dqn.train --total-steps 300 --learning-starts 32
```

## Week 4：Policy Gradient 入门、实验工程、导出

目标：先理解 VPG -> PPO 这条 on-policy 主线，再知道 Spinning Up 和导出部署在仓库中的位置。SAC、DDPG、TD3、TRPO 这周只做地图，不要求深入。

任务：

1. 阅读 `course/05_policy_gradient/index.md`、`vpg.md`、`ppo.md`。
2. 跑 VPG 和 PPO core demo。
3. 阅读 `course/06_spinningup_track.md`，只画出算法地图。
4. 跑实验快照和导出 demo。
5. 可选：只运行 SAC core demo，看连续动作和 entropy 的接口，不要求吃透。
6. 完成一次完整质量门禁。

命令：

```bash
conda run -n drl-lab python labs/05_policy_gradient/vpg/code/vpg_core_demo.py
conda run -n drl-lab python labs/05_policy_gradient/ppo/code/ppo_core_demo.py
conda run -n drl-lab python labs/07_experiment_engineering/code/snapshot_demo.py
conda run -n drl-lab python labs/08_export_deployment/code/export_demo.py
conda run -n drl-lab python -m pytest
```

可选命令：

```bash
conda run -n drl-lab python labs/05_policy_gradient/sac/code/sac_core_demo.py
```

## 月末验收

- 能说清仓库结构和学习闭环。
- 能解释 Bellman、Q-learning、DQN target、policy gradient 的基本关系。
- 至少完成四份 lab report。
- 质量门禁通过。
