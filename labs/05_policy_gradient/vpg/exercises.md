# VPG Exercises

## 纸笔题

1. 写出 VPG policy gradient objective。
2. 解释 return-to-go 和完整 episode return 的区别。
3. 说明 value baseline 为什么能降低方差。

## 代码题

1. 跑通 VPG core demo。
2. 打开 `src/drl_lab/algorithms/vpg/losses.py`，指出 policy loss。
3. 打开 `src/drl_lab/algorithms/vpg/buffer.py`，找到 advantage normalization。

## 观察题

1. advantage 标准化后均值和方差大概是多少？
2. value loss 和 policy loss 分别优化什么？

## 提交物

- 填写 `report.md`。
- 写出一次 loss shape 检查。
