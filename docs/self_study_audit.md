# Self-Study Usability Audit

本文件记录仓库成为“可独立自学 DRL Lab”的验收依据。它不是学习入口；学习入口仍是 `README.md` 和 `course/index.md`。

## 目标拆解

| 要求 | 证据 |
| --- | --- |
| 打开仓库知道怎么学 | `README.md` 是中文入口，链接 `course/index.md`，说明 `course -> labs -> src -> tests -> report` 闭环 |
| 有完整课程主线 | `course/00` 到 `course/08` 覆盖 ML、PyTorch、RL Math、Tabular、DQN、Policy Gradient、Spinning Up、Experiment、Export |
| 每章有知识点文档 | `tests/test_learning_structure.py` 检查课程章固定标题和长度下限 |
| 每个 lab 可动手 | 每个 `labs/*` 有 `notes.md`、`exercises.md`、`code/`、`report.md` |
| 练习不是只跑脚本 | lab exercises 统一包含纸笔题、代码题、观察题、提交物 |
| 报告能复盘 | lab report 统一包含目标、运行命令和“我真正理解了什么” |
| Spinning Up 正确纳入 | `course/06_spinningup_track.md`、`labs/06_spinningup_track/*/spinningup_mapping.md`、`external/spinningup/SOURCE.md`、`external/spinningup/LICENSE` |
| 现代 PyTorch 工程 | `src/drl_lab/common` 提供 seed、checkpoint、experiment、export、ONNX consistency helpers |
| 算法实现可验收 | `tests/test_dqn.py`、`test_vpg.py`、`test_ppo.py`、`test_ddpg.py`、`test_td3.py`、`test_sac.py`、`test_trpo.py` |
| 导出部署闭环 | `course/08_export_deployment.md`、`docs/export_checklist.md`、`tests/test_export_formats.py`、`tests/test_onnx_consistency.py` |
| 防止后续跑偏 | `AGENTS.md` 和 `tests/test_learning_structure.py` 约束结构、状态和外部来源 |

## 当前验收命令

```bash
conda run -n drl-lab python -m pytest
conda run -n drl-lab ruff check .
conda run -n drl-lab mypy src
```

当前课程状态使用 `usable`：学习者可以按仓库文档独立完成学习、实验、复盘和质量门禁。
