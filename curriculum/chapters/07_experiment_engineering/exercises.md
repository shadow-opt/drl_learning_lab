# Experiment Engineering Exercises

## 纸笔题

1. 写出一次可复现实验至少需要保存哪些信息。
2. 解释 config snapshot 和 checkpoint 的区别。
3. 说明为什么失败实验也要写 report。

## 代码题

1. 跑通 snapshot demo。
2. 打开 `src/drl_lab/common/experiment.py`，找到保存 config 和 environment 的函数。
3. 跑一个短 DQN 或 SAC 训练，检查 run 目录。

## 观察题

1. run 目录里有哪些文件？
2. metrics 是否能解释训练过程？
3. seed、device、dtype 是否被记录？

## 提交物

- 填写 `report.md`。
- 写出一次 run artifact 清单。
