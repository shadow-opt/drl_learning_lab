# 02 RL Math Lab Guide

## 前置阅读

先读 `course/02_rl_math.md`。本 lab 只用 GridWorld 和 value iteration，把 Bellman backup 从公式落到代码。

## 实验目标

- 跑通 value iteration demo。
- 手算一个 state 的 Bellman backup。
- 改变 discount factor，观察 value 变化。

## 代码入口

```bash
conda run -n drl-lab python labs/02_rl_math/code/value_iteration_demo.py
```

工程实现：

- `src/drl_lab/algorithms/tabular/gridworld.py`
- `src/drl_lab/algorithms/tabular/value_iteration.py`

## 提交产物

- 完成 `exercises.md`。
- 填写 `report.md`。
- 能解释 terminal state 为什么不能继续 bootstrap。

## 常见坑

- reward、next state、done 的时间顺序搞混。
- 把 Bellman expectation 和 Bellman optimality 混为一谈。
- terminal state 仍然加上未来 value。
