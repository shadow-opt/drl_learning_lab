# TRPO Exercises

## 纸笔题

1. 写出 TRPO 的 KL 约束。
2. 解释 trust region 想限制什么。
3. 说明 natural gradient 和 raw gradient 的区别。

## 代码题

1. 跑通 TRPO math demo。
2. 打开 `src/drl_lab/algorithms/trpo/math.py`，阅读 `conjugate_gradient`。
3. 修改 KL limit，观察 step scaling。

## 观察题

1. conjugate gradient residual 是否下降？
2. KL limit 变小时 step 会如何变化？

## 提交物

- 填写 `report.md`。
- 写出 PPO 与 TRPO 的关系。
