# 02 RL Math

Status: `beginner-ready`

本章是强化学习数学预备课。它用一个 4x4 GridWorld 建立 MDP、return、value、policy、Bellman equation、terminal mask 和 bootstrap 的正式语言，为 03 的 tabular RL 和 04 的 DQN 做数学铺垫。

## 先修数学

- 概率：随机变量、条件期望的直觉。
- 线性代数：向量表、矩阵显示、argmax。
- 微积分：本章几乎不用求导，但需要理解递推和固定点。
- Python：函数返回 tuple、数组 reshape、循环。

本章会从确定性 GridWorld 开始，再说明随机 MDP 中期望符号的含义。

## 预计用时

- Lecture notes：3-4 小时。
- Code walkthrough：1.5-2 小时。
- Lab：45-60 分钟。
- Problem Set：4-6 小时。
- Report：30-45 分钟。

## Lecture Map

1. MDP 五元组：state、action、transition、reward、discount。
2. Trajectory 与 return：一步 reward 和长期回报的区别。
3. Value 与 Q：状态价值和动作价值。
4. Policy：deterministic、stochastic、greedy。
5. Bellman equations：expectation、optimality、terminal mask。
6. Value iteration：backup、convergence intuition、policy extraction。

## Problem Set

完成 `exercises.md` 中至少：

- 全部概念题。
- 全部 GridWorld 手算题。
- 至少 3 道 Bellman 推导题。
- 全部代码题。
- 至少 2 道实验诊断题。

## Demo

```bash
conda run -n drl-lab python curriculum/chapters/02_rl_math/code/value_iteration_demo.py
```

## 完成标准

1. 能写出本章 GridWorld 的 MDP 元素。
2. 能区分 reward、return、value、Q value。
3. 能手算 terminal 和 non-terminal 的 Bellman backup。
4. 能读懂 values/policy 矩阵并翻译 action id。
5. 能解释 DQN target 为什么是 Bellman backup 的神经网络版本。
