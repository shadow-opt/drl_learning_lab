# ML Foundations Exercises

## 纸笔题

1. 写出线性回归的预测公式和 MSE loss。
2. 解释 learning rate 过大时 loss 为什么可能发散。
3. 用自己的话区分 underfitting 和 overfitting。
4. 说明为什么图像 batch shape 使用 `[batch, channels, height, width]`。

## 代码题

1. 跑通三个 demo：linear regression、binary classifier、image classifier。
2. 把 linear regression 的 learning rate 调大 10 倍，记录 loss 变化。
3. 找到每个 demo 写 checkpoint 和 ONNX 的代码位置。
4. 对照 `tests/test_supervised_examples.py`，说明测试验证了什么。

## 观察题

1. 每个 demo 的 loss 是否下降？如果没有，先怀疑什么？
2. checkpoint 恢复后输出是否应保持一致？为什么？
3. ONNXRuntime 和 PyTorch 输出误差是否在容差内？

## 提交物

- 填完 `report.md`。
- 贴出至少一个运行命令。
- 写下一个你故意制造并定位的失败。
