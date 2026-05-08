# 01 PyTorch Foundations

Status: `beginner-ready`

本章是 PyTorch 张量、模块、自动求导和导出的公开课式预备课。目标不是背 API，而是建立严格的 tensor contract：shape、dtype、device、gradient、module state 和 export 语义必须能被读、写、检查和解释。

## 先修数学

- 线性代数：向量、矩阵、矩阵乘法维度检查。
- 微积分：链式法则，知道复合函数如何反向传递导数。
- 概率：随机初始化和随机输入的可复现直觉。
- Python：类、函数调用、上下文管理器 `with`。

本章会用最小例题补齐 broadcasting、Jacobian-vector product 和计算图直觉。

## 预计用时

- Lecture notes：2.5-4 小时。
- Code walkthrough：1.5 小时。
- Lab：45-60 分钟。
- Problem Set：3-5 小时。
- Report：30 分钟。

## Lecture Map

1. Tensor contract：rank、shape、stride 直觉、dtype、device。
2. Shape invariants：MLP、batch 维、广播与错误广播。
3. `nn.Module`：参数注册、子模块、`forward` 调用语义。
4. Autograd：计算图、leaf tensor、chain rule、`backward`。
5. Train/eval/inference：模块状态与梯度记录分离。
6. Export：ONNX、dynamic batch、一致性检查和部署边界。

## Problem Set

完成 `exercises.md` 中至少：

- 全部概念题。
- 全部 shape 手算题。
- 至少 2 道 autograd 推导题。
- 全部代码题。
- 至少 2 道导出诊断题。

## Demo

```bash
conda run -n drl-lab python curriculum/chapters/01_pytorch_foundations/code/mlp_export_demo.py
```

## 完成标准

1. 能解释 `[3, 4] -> MLP(4, 2) -> [3, 2]` 的每一维。
2. 能说明 dtype/device mismatch 为什么出错。
3. 能解释 `model(x)`、`forward`、`loss.backward()` 的关系。
4. 能区分 `model.eval()` 和 `torch.inference_mode()`。
5. 能说明 `passed=True` 只代表 PyTorch/ONNX 输出一致，不代表模型准确。
