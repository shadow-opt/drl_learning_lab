# Roadmap

这个 roadmap 用来追踪 `drl_learning_lab` 的完整学习路线。仓库的主入口是 `README.md` 和 `course/index.md`；本文件负责说明阶段目标、产物和完成标准。

当前自学可用性验收见 `docs/self_study_audit.md`。

## 总原则

每个学习模块都必须形成学习闭环：

```text
course 讲义 -> labs 练习和 demo -> src 工程实现 -> tests 验收 -> report 复盘
```

不要只增加脚本。新增算法或主题时，必须同步补充课程说明、实验任务、报告模板、测试；如果涉及可训练神经网络，还必须补充 checkpoint、eval、导出说明和一致性检查。数学阅读模块可以没有训练入口。

## 进度矩阵

| 阶段 | 主题 | Course | Lab | Tests | Export | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| 00 | ML Foundations | `course/00_ml_foundations.md` | `labs/00_ml_foundations` | supervised tests | ONNX | usable |
| 01 | PyTorch Foundations | `course/01_pytorch_foundations.md` | `labs/01_pytorch_foundations` | checkpoint/export tests | TorchScript/ONNX | usable |
| 02 | RL Math | `course/02_rl_math.md` | `labs/02_rl_math` | tabular tests | N/A | usable |
| 03 | Tabular RL | `course/03_tabular_rl.md` | `labs/03_tabular_rl` | tabular tests | N/A | usable |
| 04 | DQN | `course/04_dqn.md` | `labs/04_dqn` | DQN tests | ONNX | usable |
| 05 | Policy Gradient | `course/05_policy_gradient` | `labs/05_policy_gradient` | algorithm tests | ONNX where applicable | usable |
| 06 | Spinning Up Track | `course/06_spinningup_track.md` | `labs/06_spinningup_track` | structure tests | N/A | usable |
| 07 | Experiment Engineering | `course/07_experiment_engineering.md` | `labs/07_experiment_engineering` | snapshot tests | artifacts | usable |
| 08 | Export Deployment | `course/08_export_deployment.md` | `labs/08_export_deployment` | export tests | ONNXRuntime | usable |

状态含义：

- `usable`：可按文档完成学习闭环。
- `complete`：内容、实验、测试、导出和外部对照都充分成熟。

## 阶段目标

### 00-01：ML / PyTorch 基础

目标：掌握后续 DRL 实现需要的监督学习、训练循环、shape、device/dtype、checkpoint、eval、export。

产物：线性回归、二分类、图像分类、MLP export demo、ONNXRuntime 一致性检查。

### 02-03：RL 数学与 Tabular RL

目标：在没有神经网络的情况下理解 MDP、return、Bellman、MC、TD、SARSA、Expected SARSA、Q-learning。

产物：GridWorld、value iteration、policy iteration、MC control、TD control、Q-learning 报告。

### 04：DQN

目标：把 tabular Q-learning 扩展到神经网络近似，理解 replay buffer、target network、TD target 和 Q-network export。

产物：DQN smoke demo、CartPole 短训练、checkpoint、eval、ONNX。

### 05-06：Spinning Up 主线

目标：学习 VPG、PPO、TRPO、DDPG、TD3、SAC，并把 Spinning Up 概念映射到本仓库现代 PyTorch 实现。

产物：每个算法的讲义、core demo、报告和测试；VPG、PPO、DDPG、TD3、SAC 有训练入口和 ONNX 导出，TRPO 当前是数学阅读模块，不提供完整 trainer。

### 07-08：实验工程与导出部署

目标：让训练结果可复现、可 debug、可导出、可独立推理。

产物：config/environment snapshot、metrics、checkpoint、`eval_result.json`、`export_report.json`、TorchScript、`torch.export`、ONNX、ONNXRuntime 一致性报告。

## 质量门禁

```bash
conda run -n drl-lab python -m pytest
conda run -n drl-lab ruff check .
conda run -n drl-lab mypy src
```
