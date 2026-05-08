# 01 代码逐行走读

本章逐行走读 `mlp_export_demo.py`，并定位它调用的公共导出入口 `src/drl_lab/common/export.py`、公共 ONNX 检查入口 `src/drl_lab/common/onnx_check.py`，以及 MLP 定义 `src/drl_lab/common/networks.py`。

## 主线

```text
seed -> MLP -> example_input -> output_path -> export_to_onnx -> compare_pytorch_onnx -> print result
```

这个 demo 不训练模型。它的目标是让你理解推理模型的输入输出、导出和一致性检查。

## `mlp_export_demo.py` 逐行走读

`from __future__ import annotations` 是类型注解兼容设置，不是学习重点。

`from pathlib import Path` 用来构造输出路径。

`import torch` 引入 PyTorch。

`from drl_lab.common.export import export_to_onnx` 引入本仓库的 ONNX 导出函数。

`from drl_lab.common.networks import MLP` 引入公共 MLP 模型。

`from drl_lab.common.onnx_check import compare_pytorch_onnx` 引入 PyTorch 与 ONNXRuntime 的输出比较函数。

`from drl_lab.common.seed import set_global_seed` 引入固定随机数的工具。

`def main() -> None:` 定义脚本入口。运行文件时真正执行的是这个函数。

`set_global_seed(7)` 固定随机性。学习点是：随机初始化的 MLP 和随机 example input 都会更容易复现。

`model = MLP(input_dim=4, hidden_sizes=[16, 16], output_dim=2)` 创建一个多层感知机。

这一行的 shape 含义：输入每个样本有 4 个特征，输出每个样本有 2 个数字，中间有两层隐藏层，每层宽度 16。

常见误解：`hidden_sizes=[16, 16]` 不是 batch size。它说的是模型内部隐藏层宽度。

`example_input = torch.randn(3, 4, dtype=torch.float32)` 创建示例输入。

这一行的 shape 含义：`3` 是 batch size，`4` 是 input_dim。dtype 是 `float32`，符合神经网络输入常规。

反例：如果写成 `torch.randn(4, 3)`，最后一维是 3，和 `input_dim=4` 对不上。

`output_path = Path("experiments/runs/mlp_export_demo/model.onnx")` 指定导出 artifact 的位置。

`export_to_onnx(model, example_input, output_path)` 执行导出。学习点是：导出需要模型和一份示例输入，这样导出工具知道输入 rank、dtype 和计算图。

`result = compare_pytorch_onnx(model, output_path, example_input)` 用同一个输入分别跑 PyTorch 和 ONNXRuntime，并比较输出差距。

`print(f"exported={output_path}")` 打印导出文件路径。

`print(f"max_abs_diff={result.max_abs_diff:.8f}")` 打印最大绝对误差。它表示所有输出元素里，PyTorch 和 ONNX 差得最多的那个差距。

`print(f"mean_abs_diff={result.mean_abs_diff:.8f}")` 打印平均绝对误差。

`print(f"passed={result.passed}")` 打印一致性检查是否通过。你最关心的是 `passed=True`。

`if __name__ == "__main__": main()` 表示直接运行脚本时调用 `main()`。

## 公共 MLP 入口

文件：`src/drl_lab/common/networks.py`

`class MLP(nn.Module)` 表示 MLP 是 PyTorch Module。

`input_dim`、`hidden_sizes`、`output_dim` 控制网络结构。

`if input_dim <= 0 or output_dim <= 0` 是参数防线，避免创建没有意义的网络。

`layers: list[nn.Module] = []` 准备按顺序收集层。

`last_dim = input_dim` 记录当前层输入宽度。

循环 `for hidden_dim in hidden_sizes` 为每个隐藏层添加 `nn.Linear(last_dim, hidden_dim)` 和激活函数。

`layers.append(nn.Linear(last_dim, output_dim))` 添加最后的输出层。

`self.net = nn.Sequential(*layers)` 把层串成一个顺序网络。

`forward` 里检查 `x.ndim != 2`。学习点是：这个 MLP 只接受 `[batch, input_dim]`。

`output = self.net(x)` 执行前向计算。

## ONNX 导出入口

文件：`src/drl_lab/common/export.py`

`export_to_onnx(model, example_input, output_path, ...)` 是本 demo 调用的导出函数。

`path.parent.mkdir(parents=True, exist_ok=True)` 确保输出目录存在。

`model.eval()` 切到推理模式。导出的是推理图，不是训练循环。

`dynamic_batch=True` 时会设置 batch 维为动态轴。学习点是：导出的模型不必只能接收 batch size 为 3 的输入，batch 维可以变化。

`with torch.inference_mode()` 禁用梯度记录。

`torch.onnx.export(...)` 真正写出 ONNX 文件。

`input_names=["input"]`、`output_names=["output"]` 给导出图命名输入输出。

`return path` 返回导出路径。

常见误解：导出函数不会训练模型，也不会评估准确率。

## ONNX 一致性检查入口

文件：`src/drl_lab/common/onnx_check.py`

`ConsistencyResult` 保存三个结果：`max_abs_diff`、`mean_abs_diff`、`passed`。

`compare_pytorch_onnx` 先 `model.eval()`，再用 PyTorch 对 `example_input` 做一次推理。

`ort.InferenceSession(...)` 加载 ONNX 文件。

`session.get_inputs()[0].name` 读取 ONNX 模型输入名。

`session.run(None, {input_name: ...})` 用 ONNXRuntime 跑同一个输入。

`diff = np.abs(torch_output - onnx_output)` 逐元素计算输出差距。

`passed=max_abs_diff <= atol` 表示最大误差不超过容忍阈值就通过。

常见误解：`passed=True` 不代表模型预测“正确答案”，只代表两个 runtime 的输出一致。

## 本章观察点

运行后你要能解释：

- `example_input` 为什么是 `[3, 4]`。
- MLP 输出为什么是 `[3, 2]`，虽然脚本没有直接打印。
- `model.onnx` 是 artifact，不是 checkpoint。
- `max_abs_diff` 和 `mean_abs_diff` 是导出一致性指标，不是训练 loss。

## 公式对应

MLP 的每个线性层对应：

```text
Y = X W^T + b
```

在 demo 中：

```text
X: [3, 4]
W_1: [16, 4]
Y_1: [3, 16]
W_2: [16, 16]
Y_2: [3, 16]
W_3: [2, 16]
Y_3: [3, 2]
```

ONNX consistency 对应数值断言：

```text
max(abs(torch_output - onnx_output)) <= 1e-5
```

这不是模型质量公式，而是 runtime 等价性检查。

## shape trace

| 位置 | shape | 说明 |
| --- | --- | --- |
| `example_input` | `[3, 4]` | 3 个样本，每个 4 个特征 |
| 第一隐藏层后 | `[3, 16]` | batch 维保留，feature 维变成 16 |
| 第二隐藏层后 | `[3, 16]` | hidden width 仍是 16 |
| 输出层后 | `[3, 2]` | 每个样本输出 2 个数字 |

公开课级读法：每经过一个 `Linear`，只允许最后一维改变，batch 维必须保持不变。
