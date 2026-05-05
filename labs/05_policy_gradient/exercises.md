# Policy Gradient Track Exercises

## 纸笔题

1. 写出 VPG policy gradient objective。
2. 解释 PPO 和 TRPO 都在限制什么。
3. 区分 stochastic policy 和 deterministic policy。
4. 说明 SAC 的 entropy bonus 解决什么问题。

## 代码题

1. 跑 `list_algorithms.py`。
2. 至少选择两个算法跑 core demo。
3. 对照两个算法的 `src/drl_lab/algorithms/<algo>/losses.py`。
4. 找到每个算法的 eval 或 export 逻辑。

## 观察题

1. 哪些算法是 on-policy？
2. 哪些算法使用 replay buffer？
3. 哪些算法有 actor 和 critic 两类网络？

## 提交物

- 填写总览 `report.md`。
- 至少完成两个算法子报告。
