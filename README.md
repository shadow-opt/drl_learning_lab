# drl_learning_lab

这是一个中文 Deep Reinforcement Learning 训练营式自学仓库。默认学习者会 Python 基础和基本命令行，但不默认会机器学习、PyTorch 或强化学习。

当前目标不是一口气冲进复杂机器人系统，而是先把以下能力打牢：

1. 机器学习和 PyTorch 基础；
2. RL 数学和 tabular RL；
3. DQN、VPG、PPO、DDPG、TD3、SAC 等经典 DRL；
4. 实验记录、debug、复现；
5. checkpoint、eval、ONNX/ONNXRuntime 导出一致性。

当前不进入 `legged_gym`、Isaac Gym、机器人控制、sim2real 或大型分布式训练。

## 怎么学

主入口是 [curriculum/README.md](curriculum/README.md)。学习内容已经重组到：

```text
curriculum/chapters/<chapter>/
  README.md       本章入口
  lesson.md       中文讲义
  walkthrough.md  手把手走读
  lab.md          实验命令、预期输出、产物解释
  exercises.md    练习
  hints.md        提示
  solutions.md    参考答案和解析
  report.md       复盘模板
  code/           教学 demo
```

不要从 `src/` 开始学。`src/drl_lab/` 是工程实现参考，只有在章节要求你对照源码时再进入。

## 当前状态

本仓库不再笼统宣称“全仓小白可自学”。章节状态见 [curriculum/status.md](curriculum/status.md)：

- `beginner-ready`：会 Python 的学习者可以按本章文档独立推进。
- `usable`：可以学习和运行，但需要更多主动查阅。
- `advanced`：适合完成基础章节后再进入。
- `reference-track`：外部资料映射，不是训练章节。
- `engineering-track`：工程专题，建议在算法基础后学习。

第一批 `beginner-ready` 章节是 `00-02`：ML Foundations、PyTorch Foundations、RL Math。

## 环境与命令

```bash
conda env create -f environment.yml
conda activate drl-lab
python -m pip install -e ".[dev]"
```

环境自检：

```bash
conda run -n drl-lab python -c "import torch, numpy, onnxruntime, gymnasium, drl_lab; print('ok')"
conda run -n drl-lab python -m pytest tests/test_learning_structure.py
```

前三章最小 demo：

```bash
conda run -n drl-lab python curriculum/chapters/00_ml_foundations/code/linear_regression.py
conda run -n drl-lab python curriculum/chapters/01_pytorch_foundations/code/mlp_export_demo.py
conda run -n drl-lab python curriculum/chapters/02_rl_math/code/value_iteration_demo.py
```

全量质量门禁：

```bash
conda run -n drl-lab python -m pytest
conda run -n drl-lab ruff check .
conda run -n drl-lab mypy src
```

## 仓库结构

```text
curriculum/         训练营式学习内容和章节状态
src/drl_lab/        现代 PyTorch 工程实现
tests/              单元测试、smoke test、导出一致性测试
docs/               debug/export checklist、实验报告模板、维护说明
external/spinningup/ Spinning Up 来源说明和 MIT license
experiments/        运行产物目录
```

## Spinning Up 的角色

OpenAI Spinning Up 是 DRL 主参考，但本仓库不会复制其旧代码风格。它作为外部概念老师使用；本仓库负责中文重写、基础补课、现代 PyTorch 实现、测试、实验产物和导出部署练习。

来源与 MIT license 保存在 [external/spinningup](external/spinningup)。
