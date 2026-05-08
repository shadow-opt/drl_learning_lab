# 02 Problem Set Hints

1. MDP 是 `(S,A,P,R,gamma)`。
2. reward 是一步，return 是多步累计。
3. Q 多了 action 维。
4. deterministic 选一个动作，stochastic 给动作分布。
5. done 时 bootstrap 为 0。
6. `state=row*cols+col`。
7. state 14 右边是 terminal 15。
8. state 13 右边不是 terminal。
9. 注意 11 不是 terminal。
10. 用 action legend。
11. reward 下标从下一步开始。
12. 固定 policy 有 `pi(a|s)`。
13. optimality 用 `max_a`。
14. 确定性转移概率只有一个 next state 为 1。
15. episode 结束后没有未来 return。
16. demo 很短，核心在 `value_iteration(env, gamma=1.0)`。
17. 搜索 `actions =`。
18. 搜索 `max` 和 `min`。
19. 搜索 `is_terminal`。
20. 搜索 `bootstrap`、`argmax`。
21. 查循环是否执行、terminal 判断、max_iterations。
22. policy 是动作 id。
23. 先影响终点邻居，再通过 backup 传播。
24. return 定义变了。
25. 正 reward 可能鼓励拖延或绕路。
