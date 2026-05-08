# 01 实验说明书

## 实验目标

运行 MLP 导出 demo，确认你能解释 shape、dtype、device、Module、forward、export 和 ONNX consistency check。这个实验没有训练循环，所以不要寻找 loss 或 optimizer。

## 运行命令

```bash
conda run -n drl-lab python curriculum/chapters/01_pytorch_foundations/code/mlp_export_demo.py
```

期望输出：

```text
exported=experiments/runs/mlp_export_demo/model.onnx
max_abs_diff=0.00000003
mean_abs_diff=0.00000001
passed=True
```

具体误差可能不同，但正常情况下 `passed=True`，误差应非常接近 0。

## 逐项解释

- `exported=...model.onnx`：ONNX 文件已经写到这个路径。
- `max_abs_diff`：PyTorch 输出和 ONNXRuntime 输出之间最大的单元素绝对差。
- `mean_abs_diff`：所有输出元素差距的平均值。
- `passed=True`：最大误差不超过公共检查函数设置的容忍阈值。

注意：这些数字不是 accuracy，不是 loss，也不说明模型“学会了任务”。本 demo 的 MLP 是随机初始化的，只用于演示导出。

## artifact

主要 artifact：

- `experiments/runs/mlp_export_demo/model.onnx`

它的用途：

- 保存一个可被 ONNXRuntime 加载的推理模型。
- 用来验证 PyTorch 模型和导出模型在同一输入上输出一致。
- 为后面导出 DQN、actor 或 critic 网络打基础。

它不包含：

- optimizer 状态。
- 训练 metrics。
- 数据集。

## 代码定位任务

打开 `curriculum/chapters/01_pytorch_foundations/code/mlp_export_demo.py`，定位：

- `MLP(input_dim=4, hidden_sizes=[16, 16], output_dim=2)`
- `torch.randn(3, 4, dtype=torch.float32)`
- `export_to_onnx(...)`
- `compare_pytorch_onnx(...)`

然后写下：

```text
example_input shape 是 [3, 4]，意思是 3 个样本，每个样本 4 个特征。
```

## 异常情况

`passed=False`：

- 先确认 `max_abs_diff` 是否明显大于 `1e-5`。
- 检查是否改过模型、导出 opset、输入 dtype。
- 检查 ONNXRuntime 版本是否异常。

shape mismatch：

- 如果你把 example input 改成 `[4, 3]`，MLP 会因为最后一维不是 4 而报错。
- 先看报错里的 `expected [batch, input_dim]`。

dtype 问题：

- 神经网络输入通常应是 `torch.float32`。
- 如果传整数 tensor，线性层一般不能按预期计算。

device mismatch：

- 如果以后把模型放到 GPU，输入也必须放到同一个 GPU。
- 本 demo 默认都在 CPU，所以正常不会遇到。

## 正常完成标准

你完成本实验，需要能回答：

1. 为什么 `example_input` 的第二维必须等于 `input_dim=4`。
2. 为什么输出路径是 artifact。
3. 为什么 `passed=True` 只说明导出一致，不说明模型准确。
4. `export_to_onnx` 和 `compare_pytorch_onnx` 分别负责什么。

## Ablation

这些改动只用于实验理解，不要提交。

| 改动 | 预期现象 | 解释 |
| --- | --- | --- |
| `example_input=torch.randn(4, 3)` | shape mismatch | 最后一维不等于 `input_dim=4` |
| `output_dim=5` | 输出 shape 变成 `[3, 5]` | batch 维不变，输出宽度改变 |
| `dtype=torch.float64` | 可能导出仍可运行但类型策略改变 | dtype 是 export contract 的一部分 |
| 关闭 dynamic batch | ONNX 输入可能固定 batch size | dynamic axes 决定部署时 batch 是否可变 |

## 诊断决策树

```text
shape mismatch？
  检查输入最后一维是否等于 input_dim
dtype 报错？
  检查神经网络输入是否为 float tensor
device mismatch？
  检查 model 和 input 是否在同一 device
passed=False？
  检查 max_abs_diff、opset、ONNXRuntime、example_input
```

## 记录表

| 项目 | 观察值 | 我的解释 |
| --- | --- | --- |
| example input shape | | |
| output shape 推断 | | |
| max_abs_diff | | |
| mean_abs_diff | | |
| passed | | |
