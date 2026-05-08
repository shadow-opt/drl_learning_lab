# 02 Problem Set

## 概念题

1. 写出有限 MDP 五元组，并解释每一项。
2. 区分 state、observation、action、reward、return。
3. 区分 `V^pi(s)`、`Q^pi(s,a)`、`V*(s)`、`Q*(s,a)`。
4. 解释 deterministic policy 和 stochastic policy。
5. 解释 terminal mask 和 bootstrap。

## 手算题

6. 画出 4x4 GridWorld state 编号矩阵。
7. 从 state 14 向右，写出 `next_state, reward, done, backup`。
8. 从 state 13 向右，假设 `values[14]=0`，手算 backup。
9. 从 state 10 向右，假设 `values[11]=0`，手算 backup。
10. policy 第一行 `[1 1 1 2]` 翻译成动作。

## 推导题

11. 写出 discounted return `G_t`。
12. 写出 Bellman expectation equation。
13. 写出 Bellman optimality equation。
14. 说明确定性 GridWorld 如何把 Bellman equation 中的求和简化掉。
15. 推导 done=True 时 target 为什么等于 reward。

## 代码题

16. 在 `value_iteration_demo.py` 中定位创建环境和运行算法的代码。
17. 在 `gridworld.py` 中定位 action id 到动作名字的顺序。
18. 在 `gridworld.py` 中定位边界裁剪逻辑。
19. 在 `value_iteration.py` 中定位 terminal skip。
20. 在 `value_iteration.py` 中定位 Bellman backup 和 greedy policy extraction。

## 实验诊断题

21. 如果 values 全是 0，可能是什么原因？
22. 如果 policy 数字被读成 value，会导致什么误解？
23. 如果 terminal 继续 bootstrap，会污染哪些 state？
24. 如果把 `gamma` 改成 0.9，values 为什么改变？
25. 如果把 `step_reward` 改成正数，会对最短路直觉产生什么影响？

## 挑战题

26. 临时把 `gamma` 改成 0.9，记录 values 差异。不要提交改动。
27. 临时创建 `terminal_reward=10` 的 GridWorld，预测并验证终点附近 values。不要提交改动。
28. 手写 state 0 的最短路径 return，并解释为什么是 -5。
29. 用一句公式连接本章 Bellman backup 和 DQN target。
30. 设计一个 RL math checklist，用于检查后续 DQN loss 是否处理 done mask。
