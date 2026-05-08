# 01 PyTorch 基础课

## 这章为什么存在

00 章里你已经看到训练循环。01 章要解决更底层的问题：PyTorch 到底在处理什么东西，为什么 shape、dtype、device 错了会直接报错或悄悄训练坏，`forward`、`backward`、`eval`、`export` 分别是什么。

后面的 DRL 代码会大量传递 observation、action、reward、done、logits、value、Q value。它们都是 tensor。你不需要一开始就会所有 PyTorch API，但必须能读懂一个 tensor 的 shape、dtype 和 device。

## tensor：PyTorch 世界里的数组

白话解释：tensor 可以先理解成更强的多维数组。一个数字是 0 维，一串数字是 1 维，表格是 2 维，图像 batch 常见是 4 维。

最小例子：

```python
import torch

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(x.shape)  # torch.Size([2, 2])
```

常见误解：tensor 不是普通 list。list 没有自动求梯度、GPU 加速、矩阵乘法和神经网络层接口。

和 DRL 的关系：环境返回的 observation 进入神经网络前通常要转成 tensor；replay buffer 采样出的 batch 也是 tensor。

## shape：每一维有多长

白话解释：shape 是 tensor 的尺寸说明。`[3, 4]` 表示 3 行 4 列；`[batch, input_dim]` 表示一批样本，每个样本有 `input_dim` 个特征。

最小例子：

```python
x = torch.randn(3, 4)
model_input_dim = 4
```

这里 `3` 是 batch size，`4` 是每个样本的特征数。如果 MLP 设置 `input_dim=4`，它能接收这个输入。

反例：如果把 `x` 写成 `torch.randn(4, 3)`，shape 是 `[4, 3]`。这不是“同样有 12 个数所以也行”，因为模型会把最后一维当成特征维，期望 4，实际 3。

常见误解：只看元素总数是不够的。神经网络层关心每一维的含义。

和 DRL 的关系：一个状态可能是 `[obs_dim]`，训练 batch 变成 `[batch, obs_dim]`；离散动作可能是 `[batch]` 的整数标签；Q 网络输出常见是 `[batch, n_actions]`。

## dtype：每个元素是什么类型

白话解释：dtype 决定 tensor 里的数字怎么存。神经网络输入和参数通常是 `float32`；分类标签经常是 `int64`，也就是 PyTorch 里的 `torch.long`。

最小例子：

```python
features = torch.randn(32, 4, dtype=torch.float32)
labels = torch.tensor([0, 1, 0], dtype=torch.long)
```

反例：`CrossEntropyLoss` 要求类别标签是 `long`，如果你传 `float32` 标签，可能直接报错。

常见误解：数字长得像 `0` 或 `1` 不代表 dtype 对。`torch.tensor([0, 1])` 和 `torch.tensor([0.0, 1.0])` 在 loss 眼里不是一回事。

和 DRL 的关系：action index 常用 `long`，reward、done mask、observation 常用 `float32`。DQN 里用 action index 从 Q 矩阵里 gather，dtype 错会立刻出问题。

## device：tensor 放在哪里

白话解释：device 表示 tensor 在 CPU 还是 GPU。模型参数和输入必须在同一设备上计算。

最小例子：

```python
device = torch.device("cpu")
model = model.to(device=device)
x = x.to(device=device, dtype=torch.float32)
```

反例：模型在 CUDA，输入在 CPU，PyTorch 会报 device mismatch。

常见误解：把模型 `.to(device)` 了，不代表之后新建的 tensor 自动在同一个 device。新 tensor 仍要显式指定或从已有 tensor 派生。

和 DRL 的关系：训练时 replay batch、policy 输入、target tensor 都要搬到同一 device。很多“代码没错但跑不起来”的问题都是 device 不一致。

## Module：可训练模型的容器

白话解释：`nn.Module` 是 PyTorch 管理神经网络的标准方式。你把层放进 Module，PyTorch 才知道哪些 parameter 要训练、保存和导出。

最小例子：

```python
from torch import nn

class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)
```

常见误解：`forward` 不是你通常直接调用的方法。习惯写 `model(x)`，PyTorch 会帮你调用 `forward`，并处理 hook 等机制。

和 DRL 的关系：后面所有网络都会是 `nn.Module`：QNetwork、PolicyNetwork、Actor、Critic。保存 checkpoint 和 ONNX 导出都依赖这个结构。

## forward：从输入到输出

白话解释：forward 是模型“怎么算预测”的路线。输入 tensor 经过层和激活函数，变成输出 tensor。

最小例子：01 demo 的 MLP 输入 shape 是 `[3, 4]`，输出 shape 是 `[3, 2]`。意思是 3 个样本，每个样本 4 个特征，模型给每个样本输出 2 个数字。

反例：如果输入 shape 是 `[4]`，缺少 batch 维，公共 `MLP.forward` 会报 `expected [batch, input_dim]`。

常见误解：forward 只负责计算输出，不负责自动更新参数。参数更新要靠 loss、backward 和 optimizer。

和 DRL 的关系：policy 的 forward 可能输出动作 logits；critic 的 forward 可能输出 value；Q 网络的 forward 可能输出每个动作的 Q value。

## autograd 和 backward：自动算梯度

白话解释：autograd 是 PyTorch 的自动求导系统。只要你的计算由 tensor 运算组成，并且参数需要梯度，PyTorch 就能从 loss 反推每个参数的 gradient。

最小例子：

```python
w = torch.tensor([1.0], requires_grad=True)
loss = (w - 3.0) ** 2
loss.backward()
print(w.grad)
```

常见误解：不是所有 tensor 都需要 gradient。训练标签、reward、done 通常不需要梯度。推理和导出检查时也不需要梯度。

和 DRL 的关系：policy gradient 里的“gradient”就是靠 autograd 算出来的；但采样环境动作、存 buffer、算 return 的很多部分不需要梯度。

## train/eval 和 inference_mode

白话解释：`model.train()` 和 `model.eval()` 切换模块状态。`torch.inference_mode()` 表示这段只推理，不记录梯度。

最小例子：

```python
model.eval()
with torch.inference_mode():
    y = model(x)
```

常见误解：`eval()` 不会帮你计算 accuracy，也不会停止 Python 代码执行。它只是告诉模型中的某些层进入推理行为。

和 DRL 的关系：训练策略和评估策略要分开。评估时不更新参数、不加探索噪声或按算法规定关闭随机性，才能得到可信指标。

## export：把模型交给推理环境

白话解释：训练好的 PyTorch 模型通常可以导出成 ONNX，让 ONNXRuntime 等环境运行。导出后要做一致性检查：同一个输入，PyTorch 和 ONNX 输出差距应该很小。

最小例子：

```text
exported=experiments/runs/mlp_export_demo/model.onnx
max_abs_diff=0.00000003
mean_abs_diff=0.00000001
passed=True
```

常见误解：导出成功不等于模型正确，也不等于训练效果好。它只说明模型图被写出去了。一致性检查才说明导出模型在示例输入上和 PyTorch 足够接近。

和 DRL 的关系：后面部署 DQN 或 actor 网络时，也会导出推理模型。训练代码可以复杂，但部署入口必须清楚输入 shape、dtype、device 和输出含义。

## 学完检查

你应该能回答：

1. tensor、shape、dtype、device 分别说明什么。
2. 为什么 `[3, 4]` 和 `[4, 3]` 不是一回事。
3. `nn.Module` 和 `forward` 的关系是什么。
4. `backward` 什么时候需要，推理时为什么不需要。
5. ONNX export 和 ONNX consistency check 分别证明什么。

## 公开课扩展：形式化定义

PyTorch 程序的核心对象可以写成：

```text
Tensor(shape, dtype, device, requires_grad)
Module(parameters, buffers, submodules, forward)
Graph(nodes=tensor ops, edges=data dependencies)
```

tensor contract 是四件事同时成立：shape 对、dtype 对、device 对、梯度语义对。只要其中一项不对，代码可能立刻报错，也可能更危险地跑出错误训练结果。

## 公开课扩展：shape invariant

MLP 的第一层 `nn.Linear(4, 16)` 接收最后一维为 4 的输入。若输入是：

```text
X: [3, 4]
W: [16, 4]
b: [16]
Y = X @ W^T + b -> [3, 16]
```

注意 PyTorch 的 `nn.Linear(in_features, out_features)` 内部权重 shape 是 `[out_features, in_features]`。这就是为什么输入最后一维必须是 `in_features`。

例题：`MLP(input_dim=4, hidden_sizes=[16, 16], output_dim=2)` 对 `example_input=[3, 4]`：

```text
[3, 4] -> Linear(4,16) -> [3,16]
[3,16] -> ReLU -> [3,16]
[3,16] -> Linear(16,16) -> [3,16]
[3,16] -> ReLU -> [3,16]
[3,16] -> Linear(16,2) -> [3,2]
```

这条 shape 推导是公开课级 PyTorch 阅读的基本功。

## 公开课扩展：broadcasting 的好与坏

broadcasting 会把较小 shape 的 tensor 自动扩展到较大 shape。它让 `Xw + b` 很方便，但也可能掩盖错误。

最小例子：

```text
pred: [32, 1]
target: [32, 1]
```

这是正确比较。危险例子：

```text
pred: [32, 1]
target: [32]
```

某些 loss 可能尝试广播，导致比较语义变成 `[32, 32]` 或报出难懂错误。课程要求你主动写 shape 注释，就是为了避免这种隐式错误。

## 公开课扩展：autograd 与链式法则

考虑复合函数：

```text
z = wx + b
a = relu(z)
L = (a - y)^2
```

链式法则告诉我们：

```text
dL/dw = dL/da * da/dz * dz/dw
```

PyTorch autograd 做的就是把每个 tensor operation 记录成计算图，再从标量 loss 反向应用链式法则。`loss.backward()` 不是魔法，它是对当前计算图做反向遍历。

常见误解：`backward()` 不知道你的任务目标，它只知道当前 loss 这个标量和产生它的 tensor 运算。loss 定义错了，梯度也会忠实地朝错误目标优化。

## 公开课扩展：Jacobian-vector product 直觉

神经网络通常有很多输出和很多参数，完整 Jacobian 巨大。PyTorch 反向传播通常不显式构造完整 Jacobian，而是高效计算 vector-Jacobian product。对入门者更重要的是直觉：

```text
标量 loss 反向传播 -> 每个参数得到一个同 shape 的 .grad
```

如果参数 `weight` shape 是 `[16, 4]`，那么 `weight.grad` 也会是 `[16, 4]`。optimizer 才能逐元素更新参数。

## 公开课扩展：Module state 与 gradient state 分离

`model.train()` 和 `model.eval()` 控制模块行为，比如 Dropout、BatchNorm。`torch.inference_mode()` 控制 autograd 是否记录计算图。二者不是同一个开关：

```text
model.eval()             -> 模型进入推理行为
torch.inference_mode()   -> 不记录梯度
```

导出时两个都需要：导出的是推理模型，并且导出过程不需要梯度。

## 公开课扩展：export contract

ONNX 导出的最低 contract 是：

```text
input name
input rank
input dtype
dynamic axes policy
output name
runtime consistency tolerance
```

`mlp_export_demo.py` 中 example input 是 `[3, 4]`，导出函数设置 dynamic batch，所以 ONNX 模型不被固定在 batch size 3，但最后一维 4 仍是结构要求。一致性检查用 `max_abs_diff <= 1e-5` 判断，不要求浮点 bit 完全相同。

## 公开课扩展：本章定理化小结

三个 PyTorch invariants：

1. Shape invariant：每个算子的输入输出 shape 必须有明确语义。
2. State invariant：训练、评估、导出要显式设置 Module 状态和梯度状态。
3. Runtime invariant：导出后的模型必须用独立 runtime 做数值一致性检查。

后续 DRL 中，batch 里会同时有 observations、actions、rewards、next observations、done masks。你能维护这些 invariants，才能可靠实现 DQN、PPO、SAC。
