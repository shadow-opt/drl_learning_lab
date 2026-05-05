# 课程索引

这个目录是 `drl_learning_lab` 的主教材。学习时不要从 `src/` 开始；先读本目录，再进入 `labs/` 做实验，最后看 `src/` 的工程实现。

## 学习闭环

大多数章节按同一套流程完成：

1. 读 `course/*.md`，理解概念、公式和算法流程。
2. 做 `labs/*/exercises.md`，至少完成纸笔题和一个代码题。
3. 跑 `labs/*/code/` 的教学 demo。
4. 对照 `src/drl_lab/` 阅读工程实现。
5. 在 `labs/*/report.md` 写复盘。
6. 跑 `pytest`、`ruff`、`mypy` 或章节指定的导出一致性检查。

例外：`06 Spinning Up Track` 是阅读映射层，不是新增训练代码章节；它的产物是 mapping、对照笔记和回到本仓库实现的索引。`02 RL Math` 和 `03 Tabular RL` 没有神经网络，不要求 ONNX。

## 路线图

| Chapter | Course | Lab | Src | Tests | Export | Status |
| --- | --- | --- | --- | --- | --- | --- |
| 00 ML Foundations | [讲义](00_ml_foundations.md) | `labs/00_ml_foundations` | supervised demos | `test_supervised_examples.py` | ONNX | usable |
| 01 PyTorch Foundations | [讲义](01_pytorch_foundations.md) | `labs/01_pytorch_foundations` | `common/*` | export/checkpoint tests | TorchScript/ONNX | usable |
| 02 RL Math | [讲义](02_rl_math.md) | `labs/02_rl_math` | tabular value tools | `test_tabular.py` | N/A | usable |
| 03 Tabular RL | [讲义](03_tabular_rl.md) | `labs/03_tabular_rl` | `algorithms/tabular` | `test_tabular.py` | N/A | usable |
| 04 DQN | [讲义](04_dqn.md) | `labs/04_dqn` | `algorithms/dqn` | `test_dqn.py` | ONNX | usable |
| 05 VPG/PPO/DDPG/TD3/SAC/TRPO | [索引](05_policy_gradient/index.md) | `labs/05_policy_gradient` | algorithm packages | algorithm tests | ONNX where applicable | usable |
| 06 Spinning Up Track | [讲义](06_spinningup_track.md) | `labs/06_spinningup_track` | all algorithms | mapping tests by structure | N/A | usable |
| 07 Experiment Engineering | [讲义](07_experiment_engineering.md) | `labs/07_experiment_engineering` | `common/experiment.py` | `test_experiment_snapshots.py` | artifacts | usable |
| 08 Export Deployment | [讲义](08_export_deployment.md) | `labs/08_export_deployment` | `common/export.py` | export tests | ONNXRuntime | usable |

状态含义：

- `usable`：可以按文档完成本阶段的学习闭环，但仍需要按 report 记录疑问和失败。
- `complete`：课程、实验、报告、测试、导出和外部对照都完整。

## 当前推荐推进

按学习依赖顺序推进：ML/PyTorch -> RL Math -> Tabular -> DQN -> VPG/PPO/TRPO -> DDPG/TD3/SAC -> Experiment Engineering -> Export Deployment。
