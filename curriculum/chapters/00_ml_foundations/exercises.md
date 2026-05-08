# 00 Problem Set

## 概念题

1. 用形式化符号写出监督学习数据集 `D`、模型 `f_theta`、loss `L` 和经验风险 `R_hat(theta)`。
2. 解释 model、parameter、prediction、target、loss、gradient、optimizer 的区别。
3. 解释为什么训练 loss 下降不等于泛化能力提升。
4. 区分 logits、probability、prediction label。
5. 解释 checkpoint、metrics、ONNX 三类 artifact 的用途。

## 手算题

6. 预测 `[2, 4]`，目标 `[1, 7]`，手算 MSE。
7. 单样本 `x=2, y=10, w=1`，loss 为 `(wx-y)^2`，手算 `dL/dw`。
8. 沿用第 7 题，学习率 `0.1`，手算一次梯度下降后的 `w`。
9. 二分类 logit 为 `0`，sigmoid 后是多少？若标签为 1，直觉上 loss 大还是小？
10. 3 类分类 logits 为 `[1, 3, 0]`，预测类别是哪一个？

## 推导题

11. 从 `L(w) = (wx-y)^2` 推导 `dL/dw = 2(wx-y)x`。
12. 对 batch MSE `R(w) = (1/N) sum_i (w x_i - y_i)^2` 写出梯度表达式。
13. 用矩阵 shape 说明为什么 `X:[N,3]`、`w:[3,1]`、`y:[N,1]` 可以计算 MSE。
14. 说明 `BCEWithLogitsLoss` 为什么不应该先手动 sigmoid。
15. 用 bias/variance/noise 框架解释三种 loss 不理想的原因。

## 代码题

16. 在 `linear_regression.py` 中定位生成真实标签的代码，写出它对应的数学公式。
17. 在 `binary_classifier.py` 中定位 accuracy 计算，说明 sigmoid 和阈值分别做什么。
18. 在 `image_classifier.py` 中定位 train/eval 切换，说明两者分别出现在哪里。
19. 列出三个 demo 中 `loss.backward()` 和 `optimizer.step()` 的位置。
20. 列出三个 demo 中 ONNX 导出和一致性检查的位置。

## 实验诊断题

21. 如果线性回归 `final_loss > 0.1`，给出三条排查路径。
22. 如果二分类 accuracy 接近 0.5，给出三条可能原因。
23. 如果图像分类 train loss 降但 val accuracy 低，如何解释？
24. 如果 ONNX consistency `passed=False`，先查什么？
25. 如果忘记 `optimizer.zero_grad()`，可能发生什么？

## 挑战题

26. 临时把线性回归 `epochs` 改为 1，记录输出并解释。不要提交改动。
27. 临时把二分类学习率改为 `0.000001`，记录输出并解释。不要提交改动。
28. 临时把图像分类每类样本数改小，观察验证指标波动。不要提交改动。
29. 用 5 句话说明本章训练循环如何迁移到 DQN。
30. 设计一个你自己的 artifact 检查清单，至少包含 metrics、checkpoint、export。
