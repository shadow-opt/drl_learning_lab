# 00 参考答案

## 概念题

1. 参考答案：`D={(x_i,y_i)}_{i=1}^N`，`f_theta(x_i)=y_hat_i`，`L(y_hat_i,y_i)` 是单样本 loss，`R_hat(theta)=(1/N)sum_i L(f_theta(x_i),y_i)` 是经验风险。评分要点：必须出现平均 loss 和参数 `theta`。为什么：训练优化的是参数化模型在训练集上的平均错误。常见错误：把 `D` 写成一个 batch。如何验证：对照 `make_dataset` 和训练循环。
2. 参考答案：model 是函数结构，parameter 是可训练数字，prediction 是模型输出，target 是监督信号，loss 是错误标量，gradient 是参数方向，optimizer 是更新规则。评分要点：区分 gradient 和 optimizer。为什么：`backward` 与 `step` 是两个动作。常见错误：说 loss 会自动更新参数。如何验证：定位 `loss.backward()` 与 `optimizer.step()`。
3. 参考答案：训练 loss 只衡量训练数据；泛化需要看验证集或新数据。评分要点：必须提到未参与训练的数据。为什么：模型可能 overfit。常见错误：loss 越低一定越好。如何验证：看图像 demo 的 `val_accuracy`。
4. 参考答案：logits 是原始分数，probability 是 sigmoid/softmax 后的概率样数值，prediction label 是阈值或 argmax 后的类别。评分要点：说明变换关系。为什么：loss 常直接用 logits。常见错误：把 logits 当概率。如何验证：读 `accuracy_from_logits`。
5. 参考答案：metrics 记录训练过程，checkpoint 保存 PyTorch 训练状态，ONNX 保存推理图并用于部署/一致性检查。评分要点：ONNX 不是 checkpoint。为什么：训练恢复和推理部署目标不同。常见错误：用 ONNX 继续训练。如何验证：看三个 demo 保存文件。

## 手算题

6. 参考答案：误差 `[1,-3]`，平方 `[1,9]`，MSE `5`。评分要点：平均而不是求和。为什么：MSE 是 mean squared error。常见错误：算成 MAE。如何验证：用 `torch.nn.MSELoss`。
7. 参考答案：`y_hat=2`，`dL/dw=2(wx-y)x=2(2-10)2=-32`。评分要点：链式法则符号正确。为什么：内层 `wx-y` 对 `w` 的导数是 `x`。常见错误：漏乘 `x`。如何验证：用 autograd。
8. 参考答案：`w_new=1-0.1*(-32)=4.2`。评分要点：减去梯度，不是加梯度。为什么：梯度下降沿负梯度方向。常见错误：符号反。如何验证：手写一行更新。
9. 参考答案：sigmoid(0)=0.5；标签为 1 时预测不够自信，loss 中等偏大。评分要点：0.5 是边界。为什么：logit 0 表示两类同等倾向。常见错误：说 logit 0 概率为 0。如何验证：`torch.sigmoid(torch.tensor(0.0))`。
10. 参考答案：类别 1，因为最大 logit 是 3，索引为 1。评分要点：用索引，不是值。为什么：分类标签是 argmax 的位置。常见错误：预测类别 3。如何验证：`torch.tensor([1,3,0]).argmax()`。

## 推导题

11. 参考答案：令 `u=wx-y`，`L=u^2`，`dL/du=2u`，`du/dw=x`，所以 `dL/dw=2(wx-y)x`。评分要点：链式法则完整。为什么：复合函数必须乘内层导数。常见错误：只写 `2(wx-y)`。如何验证：数值差分。
12. 参考答案：`dR/dw=(1/N)sum_i 2(w x_i-y_i)x_i`。评分要点：有 `1/N`。为什么：平均 loss 的梯度也是平均梯度。常见错误：忘记平均。如何验证：对 batch autograd。
13. 参考答案：`X@w` 得 `[N,1]`，和 `y:[N,1]` 同 shape，可逐元素相减平方再平均。评分要点：写出中间 shape。为什么：loss 需要 pred/target 对齐。常见错误：只说元素个数一样。如何验证：打印 demo tensor shape。
14. 参考答案：`BCEWithLogitsLoss` 内部包含稳定 sigmoid 和 BCE，手动 sigmoid 会改变数值稳定性和梯度。评分要点：提到数值稳定。为什么：极大/极小 logit 下直接 sigmoid 可能饱和。常见错误：所有分类 loss 前都先 sigmoid。如何验证：读 PyTorch loss 名称和代码用法。
15. 参考答案：训练 loss 高可能 bias 高；训练好验证差可能 variance 高；数据有噪声时最优 loss 也不为 0。评分要点：三类原因分别解释。为什么：诊断不能只看一个数字。常见错误：所有问题都归因于学习率。如何验证：比较三个 demo 指标。

## 代码题

16. 参考答案：`y = x @ true_w + true_b + noise`，对应 `y=Xw+b+epsilon`。评分要点：包括噪声项。为什么：噪声解释 loss 不为 0。常见错误：漏掉 bias。如何验证：读 `make_dataset`。
17. 参考答案：`torch.sigmoid(logits) >= 0.5` 得预测类别，再和 labels 比较均值。评分要点：sigmoid 与阈值分开说。为什么：logits 需要映射到概率样数值。常见错误：训练时也 sigmoid。如何验证：读函数。
18. 参考答案：训练 loop 中 `model.train()`；`evaluate` 和最终推理中 `model.eval()`/`inference_mode()`。评分要点：位置正确。为什么：训练和推理状态不同。常见错误：eval 自动算 accuracy。如何验证：搜索。
19. 参考答案：三个 demo 都在 loss 之后依次调用 `zero_grad`、`backward`、`step`。评分要点：顺序正确。为什么：梯度先清再算再用。常见错误：step 前没有 backward。如何验证：读训练循环。
20. 参考答案：训练结束后取 `example_input`，调用 `export_to_onnx`，再 `compare_pytorch_onnx`，失败则 raise。评分要点：说明导出后检查。为什么：artifact 要验证。常见错误：导出成功等于一致。如何验证：运行 demo。

## 实验诊断题

21. 参考答案：查 epochs/lr、数据公式、optimizer step、loss shape、seed。评分要点：至少三条。为什么：高 loss 可能是训练不足或代码错。常见错误：只增加模型深度。如何验证：做 ablation。
22. 参考答案：可能没训练、标签构造错、loss 用错、阈值错、学习率过小。评分要点：覆盖数据和优化。为什么：0.5 接近随机猜。常见错误：只怪随机种子。如何验证：看 metrics accuracy。
23. 参考答案：可能 overfit、验证集太小、数据分布不同、训练集指标不能代表泛化。评分要点：说清 train/val 差异。为什么：验证集没有参与训练。常见错误：认为代码必有 bug。如何验证：看 `metrics.csv`。
24. 参考答案：查 `max_abs_diff`、example input shape/dtype、ONNXRuntime、opset、模型是否 eval。评分要点：先看误差。为什么：passed 由最大误差阈值决定。常见错误：看 mean diff 忽略 max。如何验证：读 `compare_pytorch_onnx`。
25. 参考答案：梯度累加，更新方向混入历史 batch，可能不稳定或等效学习率异常。评分要点：提到 PyTorch 默认累加。为什么：`.grad` 不自动清空。常见错误：说会报错。如何验证：临时注释观察。

## 挑战题

26-30 参考答案应包含实验输出、改动、恢复说明和解释。评分要点：必须写“不提交临时改动”。为什么：公开课实验强调可复现和工作区清洁。常见错误：只给结论不留证据。如何验证：用 `git diff` 确认恢复。
