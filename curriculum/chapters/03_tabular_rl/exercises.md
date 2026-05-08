# Tabular RL Exercises

## 纸笔题

1. 写出 Monte Carlo control 更新式。
2. 写出 SARSA、Expected SARSA、Q-learning 更新式。
3. 解释 on-policy 和 off-policy。
4. 说明为什么 Q-learning target 使用 `max_a Q(s', a)`。

## 代码题

1. 跑通本 lab 所有 demo。
2. 修改 epsilon，比较探索强弱对 policy 的影响。
3. 修改 alpha，观察学习是否更快或更不稳定。
4. 打开 `src/drl_lab/algorithms/tabular/q_learning.py`，指出 SARSA 和 Q-learning target 差异。

## 观察题

1. VI/PI 和采样式方法得到的 policy 是否一致？
2. SARSA 是否比 Q-learning 更保守？在哪个设置下能看出来？
3. MC 和 TD 哪个更依赖完整 episode？

## 提交物

- 填完 `report.md`。
- 写出至少两个 update rule。
- 记录一次 epsilon 或 alpha 对结果的影响。
