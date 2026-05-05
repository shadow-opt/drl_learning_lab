# 04 DQN Lab Guide

## 前置阅读

先读 `course/04_dqn.md` 和 `course/03_tabular_rl.md` 中的 Q-learning 部分。

## 实验目标

- 跑通 DQN smoke demo。
- 跑一个短 CartPole 训练。
- 检查 replay buffer、TD target、target network、checkpoint 和 ONNX export。

短训练是 smoke test，目标是确认训练循环、artifact 和导出链路正常，不要求 300 steps 收敛。

## 代码入口

```bash
conda run -n drl-lab python labs/04_dqn/code/dqn_smoke_demo.py
conda run -n drl-lab python -m drl_lab.algorithms.dqn.train --total-steps 300 --learning-starts 32
```

工程实现：`src/drl_lab/algorithms/dqn`。测试：`tests/test_dqn.py`。

期望产物在 `experiments/runs/dqn_cartpole/`：

- `config.json`
- `environment.json`
- `metrics.csv`
- `q_network.pt`
- `q_network.onnx`
- `eval_result.json`
- `export_report.json`

## 提交产物

- 完成 `exercises.md`。
- 填写 `report.md`。
- 说明 DQN target 和 tabular Q-learning target 的对应关系。

## 常见坑

- replay buffer 还没满就开始采样。
- action shape 不是 `[batch]`。
- terminal state 继续 bootstrap。
- eval 时仍然使用 epsilon 探索。
