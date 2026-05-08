# 02 参考答案

## 概念题

1. 参考答案：`M=(S,A,P,R,gamma)`；分别是状态、动作、转移概率、奖励函数、折扣因子。评分要点：五项齐全。为什么：MDP 是 RL 数学对象。常见错误：漏掉 transition。如何验证：对照 GridWorld。
2. 参考答案：state 是环境真实状态；observation 是智能体看到的量；action 是选择；reward 是一步反馈；return 是未来累计回报。评分要点：reward/return 区分。为什么：长期优化不能只看一步。常见错误：reward 等于 value。如何验证：手算 state 13。
3. 参考答案：`V^pi` 是固定 policy 下 state value；`Q^pi` 是固定 policy 下 state-action value；`V*`/`Q*` 是最优值。评分要点：Q 多 action。为什么：DQN 学 Q。常见错误：V 和 Q 混用。如何验证：看 value iteration 返回 values 和 policy。
4. 参考答案：deterministic policy 每个 state 选一个 action；stochastic policy 给动作概率分布。评分要点：概率分布。为什么：policy gradient 常用随机策略。常见错误：policy 必须确定。如何验证：本章 policy 矩阵是 deterministic。
5. 参考答案：terminal mask 在 done 时把 bootstrap 设为 0；bootstrap 是用下一状态估计当前 target。评分要点：done 为 0。为什么：terminal 后无未来。常见错误：终点继续加 value。如何验证：读 `bootstrap` 行。

## 手算题

6. 参考答案：`0 1 2 3 / 4 5 6 7 / 8 9 10 11 / 12 13 14 15`。评分要点：从 0 开始。为什么：`row*cols+col`。常见错误：1 到 16。如何验证：`to_state`。
7. 参考答案：`next_state=15, reward=0, done=True, backup=0`。评分要点：terminal reward。为什么：进入终点给 0 且不 bootstrap。常见错误：reward=-1。如何验证：`transition`。
8. 参考答案：`next_state=14, reward=-1, done=False, backup=-1+1*0=-1`。评分要点：非 terminal bootstrap。为什么：14 不是 terminal。常见错误：done=True。如何验证：手查 state。
9. 参考答案：state 10 向右到 11，reward=-1，done=False，若 `values[11]=0`，backup=-1。评分要点：11 非 terminal。为什么：只有 15 是 terminal。常见错误：右边界误判。如何验证：GridWorld 编号。
10. 参考答案：state 0 right，1 right，2 right，3 down。评分要点：按 action legend 翻译。为什么：1=right，2=down。常见错误：数字当 value。如何验证：输出第一行。

## 推导题

11. 参考答案：`G_t=r_{t+1}+gamma r_{t+2}+gamma^2 r_{t+3}+...`。评分要点：折扣指数。为什么：越远 reward 权重越低。常见错误：从 `r_t` 开始。如何验证：trajectory 定义。
12. 参考答案：`V^pi(s)=sum_a pi(a|s) sum_s' P(s'|s,a)[R(s,a,s')+gamma V^pi(s')]`。评分要点：policy 和 transition 两层求和。为什么：固定 policy 下取期望。常见错误：写成 max。如何验证：lesson。
13. 参考答案：`V*(s)=max_a sum_s' P(s'|s,a)[R(s,a,s')+gamma V*(s')]`。评分要点：max over actions。为什么：最优控制选择最好动作。常见错误：仍保留 `pi`。如何验证：value iteration。
14. 参考答案：确定性环境中每个 `(s,a)` 只有一个 `s'`，所以 `sum_s'` 只剩该 next state；deterministic policy 下 `sum_a` 也只剩一个 action。评分要点：概率为 1。为什么：GridWorld transition 确定。常见错误：以为没有 P。如何验证：`transition` 返回单一结果。
15. 参考答案：done=True 表示 episode 结束，未来 reward 序列为空，所以 target 只有当前 reward。评分要点：无未来项。为什么：不能访问终点之后的 value。常见错误：加 `gamma V(terminal)`。如何验证：代码 mask。

## 代码题

16. 参考答案：`env=GridWorld()` 和 `values, policy=value_iteration(env,gamma=1.0)`。评分要点：两行都定位。为什么：环境与算法分离。常见错误：以为 demo 定义算法。如何验证：打开 demo。
17. 参考答案：`actions=("up","right","down","left")`。评分要点：顺序。为什么：policy id 依赖顺序。常见错误：自定义顺序。如何验证：action legend。
18. 参考答案：up/down/right/left 分别用 `max`/`min` 限制 row/col。评分要点：边界裁剪。为什么：不能越界。常见错误：最右列 right 一定 state+1。如何验证：读 `transition`。
19. 参考答案：`if env.is_terminal(state): continue`。评分要点：跳过 terminal。为什么：terminal 不更新。常见错误：解释 terminal policy。如何验证：value_iteration。
20. 参考答案：`bootstrap=0.0 if done else gamma*values[next_state]`，`action_values[action]=reward+bootstrap`，`policy[state]=argmax`。评分要点：backup 和 argmax。为什么：Bellman optimality。常见错误：只找 values 更新。如何验证：读核心循环。

## 实验诊断题

21. 参考答案：可能没有遍历 non-terminal、transition 全 done、max_iterations 太小、更新被跳过。评分要点：至少三种。为什么：初始 values 就是 0。常见错误：认为最优值一定全 0。如何验证：打印迭代。
22. 参考答案：会把 action id 当作分数，误以为 3 比 1 更好。评分要点：动作编号非大小。为什么：policy 是类别选择。常见错误：比较 policy 数字大小。如何验证：action legend。
23. 参考答案：先污染终点邻居的 target，再通过 value iteration 传播到更远 state。评分要点：传播。为什么：backup 递推。常见错误：只影响 terminal。如何验证：临时 ablation。
24. 参考答案：gamma 改变 return 定义，未来 reward 权重改变，因此 values 改变。评分要点：不是 bug。为什么：Bellman target 直接包含 gamma。常见错误：gamma 只影响速度。如何验证：运行 ablation。
25. 参考答案：正 step reward 可能让 agent 偏好拖延或撞墙，最短路不再显然最优，尤其 gamma=1 时可能出现无限累积问题。评分要点：奖励设计影响目标。为什么：RL 优化 reward，不优化人类直觉。常见错误：路径规划总是最短路。如何验证：改 reward 观察。

## 挑战题

26-30 参考答案应包含公式、实验输出、恢复说明和 DQN 连接。评分要点：必须说明 done mask。为什么：DQN target 与本章 Bellman backup 同构。常见错误：只写神经网络不写 Bellman。如何验证：后续 DQN loss 搜索 `done`、`gamma`、`target`。
