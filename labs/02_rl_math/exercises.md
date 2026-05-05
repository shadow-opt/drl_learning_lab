# RL Math Exercises

## 纸笔题

1. 写出 return `G_t` 的定义。
2. 写出 `V^\pi(s)` 和 `Q^\pi(s,a)` 的定义。
3. 写出 Bellman expectation equation 和 Bellman optimality equation。
4. 手算 GridWorld 中一个非终止 state 的 Bellman backup。

## 代码题

1. 跑通 value iteration demo。
2. 修改 `gamma`，观察 state value 是否更短视。
3. 打开 `src/drl_lab/algorithms/tabular/value_iteration.py`，找到 backup 对应的代码。
4. 对照 `tests/test_tabular.py`，说明测试如何判断 value iteration 正常。

## 观察题

1. value 是否会逐轮稳定？
2. terminal state 的 value 是否合理？
3. `gamma` 变小时，远期 reward 的影响如何变化？

## 提交物

- 填完 `report.md`。
- 写出一个手算 Bellman backup。
- 记录一次修改 `gamma` 的观察。
