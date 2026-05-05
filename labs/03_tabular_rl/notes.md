# 03 Tabular RL Lab Guide

## 前置阅读

先读 `course/03_tabular_rl.md`。本 lab 的重点是比较 tabular update rule，而不是追求复杂环境分数。

## 实验目标

- 跑通 VI、PI、MC control、SARSA、Expected SARSA、Q-learning demos。
- 区分 on-policy 和 off-policy。
- 比较不同 epsilon、alpha 对收敛的影响。

## 代码入口

```bash
conda run -n drl-lab python labs/03_tabular_rl/code/value_iteration_demo.py
conda run -n drl-lab python labs/03_tabular_rl/code/policy_iteration_demo.py
conda run -n drl-lab python labs/03_tabular_rl/code/monte_carlo_control_demo.py
conda run -n drl-lab python labs/03_tabular_rl/code/td_control_demo.py
conda run -n drl-lab python labs/03_tabular_rl/code/q_learning_demo.py
```

工程实现：`src/drl_lab/algorithms/tabular`。

## 提交产物

- 完成 `exercises.md`。
- 填写 `report.md`。
- 至少比较 SARSA 和 Q-learning。

## 常见坑

- SARSA 没有使用实际采样的 next action。
- Q-learning 忘记对 next state 取 max。
- terminal transition 仍然 bootstrap。
- epsilon-greedy 概率没有归一化。
