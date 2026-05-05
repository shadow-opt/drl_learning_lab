# SAC Exercises

## 纸笔题

1. 写出最大熵目标。
2. 写出 SAC critic target。
3. 解释 temperature alpha 的作用。
4. 说明 tanh-squash 为什么需要 log prob correction。

## 代码题

1. 跑通 SAC core demo。
2. 跑一个短 Pendulum 训练。
3. 打开 `src/drl_lab/algorithms/sac/networks.py`，找到 actor 的 `sample` 和 `forward`。
4. 打开 `src/drl_lab/algorithms/sac/losses.py`，指出 critic、actor、alpha loss。

## 观察题

1. log prob shape 是什么？
2. alpha 是否发生变化？
3. actor export 使用 deterministic 还是 stochastic path？

## 提交物

- 填写 `report.md`。
- 写出 SAC 与 TD3 的一个核心差异。
