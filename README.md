# drl_learning_lab

这是一个中文 Deep Reinforcement Learning 自学课程实验室。它不是算法脚本合集，也不是 Spinning Up 的搬运仓库；它的目标是让你按学习顺序完成：

1. 读懂知识点；
2. 手推关键公式或更新式；
3. 跑通教学实验；
4. 阅读现代 PyTorch 工程实现；
5. 写实验复盘；
6. 做 checkpoint、eval、ONNX/ONNXRuntime 一致性验证。

当前不进入 `legged_gym`、Isaac Gym、机器人控制、sim2real 或大型分布式训练。先把 ML、PyTorch、RL 数学、经典 DRL、实验工程和模型导出打牢。

## 怎么学

先读 [course/index.md](course/index.md)。每章都遵循同一个闭环：

```text
course/ 读讲义 -> labs/ 做练习和 demo -> src/ 看工程实现 -> tests/ 验收 -> report.md 复盘
```

推荐顺序：

1. [ML Foundations](course/00_ml_foundations.md)
2. [PyTorch Foundations](course/01_pytorch_foundations.md)
3. [RL Math](course/02_rl_math.md)
4. [Tabular RL](course/03_tabular_rl.md)
5. [DQN](course/04_dqn.md)
6. [Policy Gradient Track](course/05_policy_gradient/index.md)
7. [Spinning Up 对照学习](course/06_spinningup_track.md)
8. [Experiment Engineering](course/07_experiment_engineering.md)
9. [Export Deployment](course/08_export_deployment.md)

大多数章节按这个闭环学习；`06_spinningup_track` 是外部阅读映射层，重点是把 Spinning Up 概念映射回本仓库的中文讲义、现代 PyTorch 实现和工程检查。每章学完的最低标准是：能用自己的话解释核心概念，能写出关键公式或 update rule，能跑通对应 demo，能在 `report.md` 里记录一次失败、一次观察和一次结论。对没有神经网络的章节，不要求 ONNX 导出；对可训练神经网络，必须检查 checkpoint、eval 和 ONNX/ONNXRuntime 一致性。

## 仓库结构

```text
course/              中文课程正文和学习索引
labs/                实验作业包：导读、练习、教学 demo、报告模板
src/drl_lab/         现代 PyTorch 工程实现
tests/               单元测试、smoke test、导出一致性测试
docs/                roadmap、模板、debug/export checklist、工程规范
external/spinningup/ Spinning Up 来源说明和 MIT license
experiments/         配置、运行产物和实验报告位置
```

`course` 是学习入口；`labs` 是动手区；`src` 是工程参考。不要从 `src` 直接开始学算法，否则会被工程细节淹没。

## 环境与命令

推荐使用 conda：

```bash
conda env create -f environment.yml
conda activate drl-lab
python -m pip install -e ".[dev]"
```

质量门禁：

```bash
conda run -n drl-lab python -m pytest
conda run -n drl-lab ruff check .
conda run -n drl-lab mypy src
```

环境自检：

```bash
conda run -n drl-lab python -c "import torch, numpy, onnxruntime, gymnasium, drl_lab; print('ok')"
conda run -n drl-lab python -m pytest tests/test_learning_structure.py
```

常用学习 demo：

```bash
conda run -n drl-lab python labs/00_ml_foundations/code/linear_regression.py
conda run -n drl-lab python labs/01_pytorch_foundations/code/mlp_export_demo.py
conda run -n drl-lab python labs/02_rl_math/code/value_iteration_demo.py
conda run -n drl-lab python labs/03_tabular_rl/code/q_learning_demo.py
conda run -n drl-lab python labs/04_dqn/code/dqn_smoke_demo.py
conda run -n drl-lab python -m drl_lab.algorithms.dqn.train --total-steps 300 --learning-starts 32
conda run -n drl-lab python labs/08_export_deployment/code/export_demo.py
```

这些短训练和 demo 是 smoke test，用来确认代码、shape、artifact 和导出链路正常；它们不代表算法已经在 benchmark 上收敛。

更多命令见各章 `course/*.md` 和对应 `labs/*/notes.md`。

## 自学可用性审计

仓库的课程结构、lab 闭环、Spinning Up 来源、AGENTS 护栏和质量门禁由 [docs/self_study_audit.md](docs/self_study_audit.md) 记录，并由 `tests/test_learning_structure.py` 做结构检查。

## Spinning Up 的角色

OpenAI Spinning Up 是 DRL 主参考，但本仓库不会直接复制其旧代码风格。每个相关算法章都会说明：

- Spinning Up 对应阅读位置；
- 本仓库吸收了哪些概念；
- 现代 PyTorch 实现在哪里；
- 哪些工程能力是本仓库额外补充的。

来源与 MIT license 保存在 [external/spinningup](external/spinningup)。

## 贡献和维护

贡献者和 AI agent 必须先读 [AGENTS.md](AGENTS.md)。核心原则：新增内容必须服务学习闭环。不要只加脚本，不要堆空文件，不要扩展到当前范围外的大系统。
