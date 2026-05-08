# 01 参考答案

## 概念题

1. 参考答案：tensor contract 包含 shape、dtype、device、requires_grad。评分要点：四项齐全。为什么：任一项错都可能破坏计算。常见错误：只检查 shape。如何验证：打印 tensor 属性。
2. 参考答案：rank 是维度数量；batch dim 是样本批维；feature dim 是每个样本的特征维。评分要点：说明 `[batch, feature]`。为什么：MLP 依赖最后一维。常见错误：把 feature dim 当 batch。如何验证：`example_input.shape`。
3. 参考答案：Module 是容器；parameter 会被 optimizer 更新；buffer 随模型保存但通常不训练；submodule 是嵌套模块。评分要点：parameter/buffer 区分。为什么：checkpoint/export 依赖注册。常见错误：普通属性也会自动训练。如何验证：`model.parameters()`。
4. 参考答案：`train/eval` 控制模块行为；`inference_mode` 禁用梯度记录。评分要点：二者分离。为什么：推理既要稳定状态也不需要图。常见错误：eval 自动禁用梯度。如何验证：读 export。
5. 参考答案：export 写 ONNX 推理图；consistency check 比较 PyTorch 和 ONNXRuntime 输出。评分要点：导出不等于正确。为什么：部署要验证 runtime 等价。常见错误：passed=True 当 accuracy。如何验证：运行 demo。

## 手算题

6. 参考答案：rank 2，shape `[3,4]`，batch 3，feature 4。评分要点：rank 正确。为什么：有两个轴。常见错误：batch 4。如何验证：打印 shape。
7. 参考答案：`[3,16]`。评分要点：batch 保留。为什么：Linear 改最后一维。常见错误：`[16,3]`。如何验证：临时运行。
8. 参考答案：`[3,4] -> [3,16] -> [3,16] -> [3,2]`，ReLU 不改 shape。评分要点：包含输出层。为什么：每层线性只改 feature。常见错误：hidden size 当层数。如何验证：打印。
9. 参考答案：可能广播或 loss 语义错，应显式对齐为 `[32,1]` 或按 loss 要求改成 `[32]`。评分要点：说出广播风险。为什么：shape 相近不等于语义一致。常见错误：元素数差不多就行。如何验证：构造小例子。
10. 参考答案：`[16,4]`。评分要点：grad 与参数同 shape。为什么：optimizer 逐元素更新。常见错误：写成输入 shape。如何验证：反向后看 `.grad.shape`。

## 推导题

11. 参考答案：`X:[B,4]`，`W:[16,4]`，`W^T:[4,16]`，`XW^T:[B,16]`，`b:[16]` broadcast 到 `[B,16]`。评分要点：转置和 bias。为什么：`nn.Linear` 权重按 `[out,in]` 存。常见错误：漏转置。如何验证：读 PyTorch docs 或打印参数 shape。
12. 参考答案：`dL/dw=2(z-y)x`。评分要点：链式法则。为什么：`dz/dw=x`。常见错误：漏 bias 对 w 无影响。如何验证：autograd。
13. 参考答案：标量 loss 对每个参数元素都有偏导，组合成与参数同 shape 的梯度 tensor。评分要点：同 shape。为什么：更新规则需要一一对应。常见错误：所有参数共享一个梯度。如何验证：查看 `.grad`。
14. 参考答案：dynamic batch 允许样本数变化，但模型第一层权重固定要求 feature dim 为 4。评分要点：batch 和 feature 区分。为什么：权重矩阵维度固定。常见错误：以为任意 shape 都能输入。如何验证：ONNX dynamic axes。
15. 参考答案：`pred:[32,1]` 和 `target:[32]` 可能触发错误广播；应明确 reshape。评分要点：给具体 shape。为什么：隐式扩展可能改变比较矩阵。常见错误：依赖 PyTorch 猜。如何验证：运行 loss。

## 代码题

16. 参考答案：`torch.randn(3,4,dtype=torch.float32)`；float32 是神经网络输入常用 dtype。评分要点：路径和 dtype。为什么：Linear 参数默认 float32。常见错误：忽略 dtype。如何验证：打印。
17. 参考答案：`if x.ndim != 2`。评分要点：说明期望 `[batch,input_dim]`。为什么：公共 MLP 只接二维输入。常见错误：单样本 `[4]` 直接输入。如何验证：读文件。
18. 参考答案：`export_to_onnx` 中先 `model.eval()`，再 `with torch.inference_mode()` 包住 `torch.onnx.export`。评分要点：两个位置。为什么：导出推理图。常见错误：只提 eval。如何验证：读函数。
19. 参考答案：`passed=max_abs_diff <= atol`。评分要点：max 不是 mean。为什么：最大单点误差必须受控。常见错误：平均小就通过。如何验证：读返回值。
20. 参考答案：`experiments/runs/mlp_export_demo/model.onnx`，用于 ONNXRuntime 推理和一致性检查。评分要点：用途明确。为什么：artifact 服务部署。常见错误：当训练 checkpoint。如何验证：运行 demo。

## 实验诊断题

21. 参考答案：`[4,3]` 最后一维 3 不匹配 `input_dim=4`。评分要点：最后一维。为什么：Linear 内维对不上。常见错误：说 batch 从 3 变 4 所以错。如何验证：临时改动。
22. 参考答案：仍需关注 `max_abs_diff`，因为 passed 由最大误差决定。评分要点：max。为什么：少数输出元素可能严重偏离。常见错误：只看 mean。如何验证：读 check。
23. 参考答案：CPU tensor 和 CUDA tensor 位于不同设备，同一矩阵乘法不能跨设备。评分要点：同一算子。为什么：PyTorch 不隐式搬运。常见错误：模型 to cuda 就够。如何验证：看 00 图像 demo `.to(device)`。
24. 参考答案：`CrossEntropyLoss` 要求 label 为 long；BCE 常要求 float target。评分要点：不同 loss 要求不同。为什么：类别索引和概率目标语义不同。常见错误：所有 label 都 float。如何验证：读 demo。
25. 参考答案：Dropout/BatchNorm 等层可能保留训练行为，导出图不稳定或不符合推理语义。评分要点：潜在风险。为什么：Module state 影响 forward。常见错误：当前 MLP 没这些层所以永远无所谓。如何验证：看 export 习惯。

## 挑战题

26-30 参考答案应包含临时改动、观察输出、shape 推断、恢复说明。评分要点：证据和恢复。为什么：公开课实验强调可复现。常见错误：留下未提交改动。如何验证：`git diff`。
