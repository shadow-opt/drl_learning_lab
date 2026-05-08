# 00 Problem Set Hints

1. `D={(x_i,y_i)}`，经验风险是训练集平均 loss。
2. 按“谁是函数、谁是数字、谁是更新规则”分类。
3. 想训练集和验证集的差异。
4. logits 是原始分数，probability 是变换后的 0 到 1 数值。
5. metrics 看过程，checkpoint 恢复模型，ONNX 做推理部署。
6. 先算误差，再平方，再平均。
7. 用链式法则。
8. `w_new = w_old - lr * gradient`。
9. sigmoid(0) 是中点。
10. 取最大 logit 的索引。
11. 把 `wx-y` 看成中间变量。
12. batch 梯度是单样本梯度平均。
13. 关注矩阵乘法内维是否相等。
14. 这个 loss 名字里已经包含 logits。
15. bias 是模型太弱，variance 是太依赖训练集，noise 是数据不可解释扰动。
16. 找 `true_w`、`true_b` 和 `x @ true_w`。
17. 找 `accuracy_from_logits`。
18. 搜索 `model.train()` 和 `model.eval()`。
19. 三个 demo 的训练循环结构相同。
20. 搜索 `export_to_onnx` 和 `compare_pytorch_onnx`。
21. 先查 lr、epochs、数据公式和 optimizer step。
22. 可能是没训练、标签错、阈值或 loss 用错。
23. 这是泛化诊断题，不只是运行错误。
24. 先看最大误差和 example input。
25. PyTorch 梯度默认累加。
