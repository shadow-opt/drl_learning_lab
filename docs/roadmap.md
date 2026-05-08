# Roadmap

本 roadmap 追踪 `drl_learning_lab` 的训练营式学习路线。主入口是 `README.md` 和 `curriculum/README.md`。

## 总原则

学习内容统一放在 `curriculum/chapters/<chapter>/`。每章至少包含：

```text
README.md -> lesson.md -> walkthrough.md -> lab.md -> exercises.md -> hints.md -> solutions.md -> report.md
```

算法和工程实现仍放在 `src/drl_lab/`，测试仍放在 `tests/`。

## 进度矩阵

| 阶段 | 主题 | Curriculum | Tests | 状态 |
| --- | --- | --- | --- | --- |
| 00 | ML Foundations | `curriculum/chapters/00_ml_foundations` | supervised tests | beginner-ready |
| 01 | PyTorch Foundations | `curriculum/chapters/01_pytorch_foundations` | checkpoint/export tests | beginner-ready |
| 02 | RL Math | `curriculum/chapters/02_rl_math` | tabular tests | beginner-ready |
| 03 | Tabular RL | `curriculum/chapters/03_tabular_rl` | tabular tests | usable |
| 04 | DQN | `curriculum/chapters/04_dqn` | DQN tests | usable |
| 05 | Policy Gradient | `curriculum/chapters/05_policy_gradient` | algorithm tests | advanced |
| 06 | Spinning Up Track | `curriculum/chapters/06_spinningup_track` | structure tests | reference-track |
| 07 | Experiment Engineering | `curriculum/chapters/07_experiment_engineering` | snapshot tests | engineering-track |
| 08 | Export Deployment | `curriculum/chapters/08_export_deployment` | export tests | engineering-track |

状态定义见 `curriculum/status.md`。

## 下一批工作

1. 把 `03_tabular_rl` 升级到 `beginner-ready`。
2. 把 `04_dqn` 拆成 tabular Q-learning 到 neural Q-learning 的细 walkthrough。
3. 再拆 `05_policy_gradient`：先 VPG/PPO 入门，再连续控制。
