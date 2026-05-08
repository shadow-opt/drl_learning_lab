# Curriculum

这里是 `drl_learning_lab` 的主学习入口。默认你会 Python 基础，但不默认你会机器学习、PyTorch 或强化学习。

## 学习方法

每章按同一套训练营闭环推进：

1. `README.md`：确认本章目标、先修和完成标准。
2. `lesson.md`：读知识点讲解。
3. `walkthrough.md`：跟着最小路径走一遍。
4. `lab.md`：跑 demo，核对预期输出和产物。
5. `exercises.md`：做必做题；卡住先看 `hints.md`。
6. `solutions.md`：对照参考答案，修正理解。
7. `report.md`：写复盘，记录失败和下一步。

不要直接从 `src/` 或测试文件开始。那些是工程参考和验收工具，不是第一入口。

## 推荐顺序

| 顺序 | 章节 | 状态 | 说明 |
| --- | --- | --- | --- |
| 00 | [ML Foundations](chapters/00_ml_foundations/README.md) | beginner-ready | loss、gradient、训练循环 |
| 01 | [PyTorch Foundations](chapters/01_pytorch_foundations/README.md) | beginner-ready | tensor、shape、Module、autograd |
| 02 | [RL Math](chapters/02_rl_math/README.md) | beginner-ready | MDP、return、value、Bellman backup |
| 03 | [Tabular RL](chapters/03_tabular_rl/README.md) | usable | MC、TD、SARSA、Q-learning |
| 04 | [DQN](chapters/04_dqn/README.md) | usable | replay buffer、target network、TD target |
| 05 | [Policy Gradient Track](chapters/05_policy_gradient/README.md) | advanced | VPG、PPO、TRPO、DDPG、TD3、SAC |
| 06 | [Spinning Up Track](chapters/06_spinningup_track/README.md) | reference-track | 外部资料映射 |
| 07 | [Experiment Engineering](chapters/07_experiment_engineering/README.md) | engineering-track | 实验管理和复现 |
| 08 | [Export Deployment](chapters/08_export_deployment/README.md) | engineering-track | ONNX、ONNXRuntime、一致性测试 |

完整状态说明见 [status.md](status.md)。术语卡片见 [glossary.md](glossary.md)。

## 前三章最小路径

```bash
conda run -n drl-lab python curriculum/chapters/00_ml_foundations/code/linear_regression.py
conda run -n drl-lab python curriculum/chapters/01_pytorch_foundations/code/mlp_export_demo.py
conda run -n drl-lab python curriculum/chapters/02_rl_math/code/value_iteration_demo.py
```

如果这些命令跑不通，先不要继续 DQN 或 PPO。回到对应章节的 `lab.md` 和 `hints.md` 排查。
