# Chapter Progress Plan

这个文件替代旧的 first month plan。学习节奏按章节闯关，不按固定日历推进。

## Gate 0：环境

```bash
conda run -n drl-lab python -c "import torch, numpy, onnxruntime, gymnasium, drl_lab; print('ok')"
conda run -n drl-lab python -m pytest tests/test_learning_structure.py
```

## Gate 1：00-02 beginner-ready

```bash
conda run -n drl-lab python curriculum/chapters/00_ml_foundations/code/linear_regression.py
conda run -n drl-lab python curriculum/chapters/01_pytorch_foundations/code/mlp_export_demo.py
conda run -n drl-lab python curriculum/chapters/02_rl_math/code/value_iteration_demo.py
```

完成标准：

- 三章 report 都填写。
- 能解释 loss、shape、Bellman backup。
- 知道 artifact 是什么，但不需要深究 ONNX。

## Gate 2：03-04 usable

进入 tabular RL 和 DQN。这里开始需要更强主动性；如果看不懂，回到 02。

## Gate 3：05+

完成 DQN 后再进入 VPG/PPO。DDPG/TD3/SAC 和导出部署不要抢跑。
