# DDPG Exercises

## 纸笔题

1. 写出 DDPG critic target。
2. 写出 deterministic actor objective。
3. 解释为什么 DDPG 训练时需要 exploration noise。

## 代码题

1. 跑通 DDPG core demo。
2. 跑一个短 Pendulum 训练。
3. 打开 `src/drl_lab/algorithms/ddpg/losses.py`，指出 critic loss 和 actor loss。
4. 找到 soft target update。

## 观察题

1. actor 输出 action shape 是什么？
2. critic 输入为什么同时需要 obs 和 action？
3. actor/critic ONNX export 是否生成？

## 提交物

- 填写 `report.md`。
- 写出一次连续动作 shape 检查。
