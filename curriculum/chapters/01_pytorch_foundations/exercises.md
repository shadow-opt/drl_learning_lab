# 01 Problem Set

## 概念题

1. 定义 tensor contract：shape、dtype、device、requires_grad。
2. 解释 rank、batch dim、feature dim。
3. 解释 `nn.Module`、parameter、buffer、submodule 的区别。
4. 区分 `model.train()`、`model.eval()`、`torch.inference_mode()`。
5. 解释 ONNX export 和 consistency check 的目标。

## 手算题

6. 对 `torch.randn(3,4)` 写出 rank、shape、batch size、feature dim。
7. `Linear(4,16)` 接收 `[3,4]`，输出 shape 是什么？
8. `MLP(4,[16,16],2)` 接收 `[3,4]`，逐层写 shape。
9. 如果 target shape 是 `[32]`，pred shape 是 `[32,1]`，为什么要警惕？
10. 参数 weight shape `[16,4]`，它的 grad shape 应是什么？

## 推导题

11. 写出线性层 `Y=XW^T+b` 的 shape 推导。
12. 用链式法则解释 `z=wx+b, L=(z-y)^2` 如何得到 `dL/dw`。
13. 说明为什么标量 loss 反向传播后每个 parameter 得到同 shape grad。
14. 解释 dynamic batch 只放开 batch 维，不放开 feature 维。
15. 用一个例子说明 broadcasting 可能掩盖 shape bug。

## 代码题

16. 在 `mlp_export_demo.py` 中定位 example input，并解释 dtype。
17. 在 `src/drl_lab/common/networks.py` 中定位 MLP shape 检查。
18. 在 `src/drl_lab/common/export.py` 中定位 `model.eval()` 和 `inference_mode()`。
19. 在 `src/drl_lab/common/onnx_check.py` 中定位 `passed` 的计算。
20. 写出 demo 中 ONNX artifact 的路径和用途。

## 实验诊断题

21. 如果输入改成 `[4,3]`，为什么会失败？
22. 如果 `passed=False` 但 `mean_abs_diff` 很小，仍需关注什么？
23. 如果模型在 CUDA、输入在 CPU，为什么会失败？
24. 如果把分类 label dtype 写成 float，哪些 loss 会受影响？
25. 如果忘记 `model.eval()` 就导出，潜在风险是什么？

## 挑战题

26. 临时打印 MLP 每层输出 shape。不要提交改动。
27. 临时把 `output_dim` 改成 5，推断并验证输出 shape。不要提交改动。
28. 临时关闭 dynamic batch，观察导出模型 metadata。不要提交改动。
29. 设计一个 PyTorch shape checklist，至少 6 项。
30. 用 5 句话说明 01 的 tensor contract 如何迁移到 replay buffer batch。
