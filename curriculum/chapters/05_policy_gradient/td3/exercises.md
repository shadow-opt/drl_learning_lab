# TD3 Exercises

## 纸笔题

1. 写出 TD3 critic target。
2. 解释 clipped double Q-learning。
3. 解释 target policy smoothing 和 delayed policy update。

## 代码题

1. 跑通 TD3 core demo。
2. 跑一个短 Pendulum 训练。
3. 打开 `src/drl_lab/algorithms/td3/losses.py`，找到 min-Q target。
4. 找到 actor 延迟更新逻辑。

## 观察题

1. 两个 critic 输出是否 shape 一致？
2. target noise 裁剪范围是多少？
3. actor export 是否和 DDPG 类似？

## 提交物

- 填写 `report.md`。
- 写出 TD3 三项 trick 的作用。
