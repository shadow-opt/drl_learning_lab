# 01 PyTorch Foundations

## 学习目标

本章训练 PyTorch 工程能力。学完后你应该能稳定处理 tensor shape、device、dtype、autograd、`nn.Module`、checkpoint、eval、TorchScript、`torch.export`、ONNX 和 ONNXRuntime 一致性测试。

这不是“会用 PyTorch API”那么简单。Deep RL 训练循环经常同时包含环境交互、replay buffer、多个网络、target network、随机 action、eval policy 和导出图。没有扎实的 PyTorch 工程习惯，算法公式即使懂了也很难落地。

## 为什么重要

RL 中的 bug 往往不会直接报错。比如 action shape 多一维，代码仍然能跑；done mask dtype 错了，loss 也能算；target network 没有停止梯度，训练会悄悄变不稳定。PyTorch 基础章的目标就是建立一套“先检查工程不变量”的习惯。

本仓库要求所有可训练模型最终都能导出 ONNX，并用 ONNXRuntime 做一致性测试。这会强迫你把训练图和推理图分清楚，也会暴露动态控制流、随机采样、shape 不稳定等问题。

## 核心直觉

Tensor 是带 shape 和 dtype 的数值容器。`nn.Module` 是参数和计算的组织方式。Autograd 记录计算图并求梯度。Optimizer 只负责根据参数的 `.grad` 更新参数。

训练时你关心梯度；推理、评估、导出时你关心确定性和稳定 I/O。很多代码需要在 train/eval 两种模式下表现不同，例如 dropout 和 batch norm。RL policy 还可能在训练时采样 action，在导出时输出 deterministic action。

本仓库里要特别注意“训练函数”和“推理函数”不是同一个概念。SAC actor 的 `sample()` 服务训练 loss，需要随机性和 log probability；`forward()` 服务 eval/export，需要稳定 action。DQN Q-network 的 `forward()` 可以直接导出，因为它本身就是 deterministic inference。写新模型时，先问清楚：这个方法是否需要梯度、是否包含随机采样、是否会被导出。

## 数学与公式

反向传播计算的是链式法则：

$$
\frac{\partial L}{\partial w} =
\frac{\partial L}{\partial y}
\frac{\partial y}{\partial w}
$$

你不需要手写这个过程，但必须知道：只有参与计算图的 tensor 才能传回梯度；`detach()` 会切断梯度；`torch.no_grad()` 和 `torch.inference_mode()` 会禁止记录计算图。

对于神经网络输入输出，本仓库使用显式 shape 约定：

```text
obs:     [batch, obs_dim]
action:  [batch] or [batch, action_dim]
reward:  [batch]
done:    [batch]
logits:  [batch, action_dim]
value:   [batch]
q_value: [batch, action_dim] or [batch, 1]
```

## 算法流程

一个可靠 PyTorch 模块开发流程：

1. 先写 `forward` 的 shape 注释。
2. 用一个小 batch 做 smoke test。
3. 检查输出 shape、dtype、device。
4. 构造一个标量 loss，确认 `backward()` 后参数有梯度。
5. 保存 checkpoint，再加载并比较输出。
6. 导出 ONNX 或 TorchScript。
7. 用同一输入比较 PyTorch 与导出模型输出。

## 代码双索引

- 教学 demo：`labs/01_pytorch_foundations/code/mlp_export_demo.py`
- 网络工具：`src/drl_lab/common/networks.py`
- checkpoint：`src/drl_lab/common/checkpoint.py`
- export：`src/drl_lab/common/export.py`
- ONNXRuntime 比较：`src/drl_lab/common/onnx_check.py`
- 测试：`tests/test_checkpoint.py`、`tests/test_export_formats.py`、`tests/test_onnx_consistency.py`

## 实验任务

运行：

```bash
conda run -n drl-lab python labs/01_pytorch_foundations/code/mlp_export_demo.py
```

然后阅读 `src/drl_lab/common/export.py`，确认 TorchScript、`torch.export` 和 ONNX 的入口分别是什么。把 batch size 从 1 改成 4，观察导出一致性测试是否仍然成立。

## Debug Checklist

- 模型参数和输入是否在同一 device。
- float 输入是否误用了 integer dtype。
- loss 是否是 scalar。
- 是否在 target 计算中漏了 `torch.no_grad()`。
- 导出模型是否包含随机采样。
- ONNX 输入名、输出名和动态 batch 维是否正确。
- checkpoint 恢复后，模型参数、optimizer state 和关键 config 是否能对应。
- batch size 从 1 改到 4 时，输出 shape 是否只在 batch 维变化。

## Spinning Up 对照

Spinning Up 的旧 PyTorch/TensorFlow 风格更偏教学实现，不覆盖现代 `torch.export`、ONNXRuntime 一致性测试和严格类型检查。本章是本仓库对 Spinning Up 的工程补充。

## 学完标准

- 能解释 train/eval、no_grad/inference_mode、detach 的区别。
- 能写带 shape 注释的 `nn.Module`。
- 能保存和恢复 checkpoint。
- 能导出一个小模型并比较 PyTorch 与 ONNXRuntime 输出。
- 能说明训练路径和导出路径为什么要分开。
- 能用一个小 batch 独立验证 forward/backward/export。
- 能在 `labs/01_pytorch_foundations/report.md` 记录一次导出实验。
