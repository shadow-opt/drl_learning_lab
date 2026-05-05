# 08 Export Deployment

## 学习目标

本章目标是让训练好的模型离开训练循环独立推理。学完后你应该能解释 TorchScript、`torch.export`、ONNX、ONNXRuntime 的角色，能为 policy、value function、Q-network、actor、critic 设计 inference input/output，并能做 PyTorch vs ONNXRuntime 一致性测试。

本仓库要求：每个可训练模型最终都应该有导出路径。能训练不等于能部署；能部署至少要有稳定推理图、明确输入输出、可重复导出和数值一致性验证。

## 为什么重要

RL 训练代码里常常混有环境交互、随机采样、探索噪声、replay buffer、loss 计算和 optimizer。部署时只需要 inference path。如果你没有主动分离训练和推理，导出时很容易把随机采样或训练专用逻辑带进去。

导出还能倒逼模型接口设计。比如 DQN Q-network 输入 `obs` 输出 `[batch, action_dim]`；DDPG critic 输入 `obs/actions` 输出 `[batch]`；SAC actor 训练时 `sample()`，导出时 `forward()` deterministic action。

## 核心直觉

导出部署可以理解为三步：

1. 固定一个 inference function。
2. 用 example input 捕获计算图。
3. 用另一个 runtime 验证输出一致。

如果 ONNXRuntime 和 PyTorch 对同一输入输出差距很大，说明导出图、输入 shape、dtype、模型模式或随机路径可能有问题。

四个名字的分工：

| 名称 | 在本仓库的用途 | 适合检查什么 |
| --- | --- | --- |
| TorchScript | PyTorch 内部可保存推理模块 | trace 后是否仍匹配 PyTorch |
| `torch.export` | PyTorch 2.x 的图捕获与导出中间形态 | inference graph 是否能被稳定捕获 |
| ONNX | 跨 runtime 的交换格式 | 算法模型能否离开 PyTorch 训练代码 |
| ONNXRuntime | 独立推理 runtime | PyTorch vs ONNX 数值是否一致 |

算法章节的主要交付是 ONNX + ONNXRuntime 一致性；TorchScript 和 `torch.export` 主要在本章作为通用工程能力练习。

## 数学与公式

一致性测试通常检查最大绝对误差：

$$
\max_i |y_i^{torch} - y_i^{onnx}| < \epsilon
$$

也可以记录平均绝对误差：

$$
\frac{1}{N}\sum_i |y_i^{torch} - y_i^{onnx}|
$$

容差要结合 dtype 和算子设置。这个仓库的目标不是证明两个 runtime 数学完全相同，而是确保导出图在工程容差内等价。

## 算法流程

一次 ONNX 导出流程：

1. `model.eval()`。
2. 构造固定 example input。
3. 用 `torch.inference_mode()` 得到 PyTorch reference output。
4. 调用导出函数，设置 input/output names 和 dynamic batch。
5. 用 ONNXRuntime 加载模型。
6. 用同一输入运行 ONNXRuntime。
7. 比较 `max_abs_diff` 和 `mean_abs_diff`。
8. 把导出路径和误差写入 report。

Multi-input critic 要显式命名输入，例如 `obs` 和 `actions`，否则后续部署很难可靠调用。

导出前还要决定“部署时到底需要哪个网络”。DQN 需要 Q-network；VPG/PPO 通常需要 policy，有时也导出 value function 方便分析；DDPG/TD3/SAC 至少需要 actor，critic 通常用于验证和离线评估。训练时的 optimizer、replay buffer、loss function 不属于 inference graph。

## 代码双索引

- Demo：`labs/08_export_deployment/code/export_demo.py`
- Export helpers：`src/drl_lab/common/export.py`
- ONNXRuntime checks：`src/drl_lab/common/onnx_check.py`
- DQN export：`src/drl_lab/algorithms/dqn/export_onnx.py`
- SAC actor export path：`src/drl_lab/algorithms/sac/networks.py`
- Export tests：`tests/test_export_formats.py`
- ONNX consistency tests：`tests/test_onnx_consistency.py`
- Checklist：`docs/export_checklist.md`

## 实验任务

```bash
conda run -n drl-lab python labs/08_export_deployment/code/export_demo.py
```

期望看到类似输出：

```text
onnx: path=experiments/runs/export_demo/model.onnx passed=True max_abs_diff=0.00000000 mean_abs_diff=0.00000000
torchscript: path=experiments/runs/export_demo/model.ts passed=True ...
torch_export: path=experiments/runs/export_demo/model.pt2 passed=True ...
```

然后选择一个算法训练入口，例如 DQN 或 SAC，检查 run 目录里的 ONNX artifact。在 `labs/08_export_deployment/report.md` 记录：

- 模型类型；
- 输入名和输出名；
- input shape；
- output shape；
- PyTorch vs ONNXRuntime 最大误差。
- `export_report.json` 中的 `input_names`、`output_names`、`input_shape`、`output_shape` 是否和你理解一致。

## Debug Checklist

- 是否在导出前调用 `model.eval()`。
- 是否使用 `torch.inference_mode()` 计算 reference。
- example input dtype/device 是否正确。
- dynamic batch 维是否设置。
- stochastic actor 是否使用 deterministic `forward()`。
- multi-input critic 是否显式命名输入。
- 导出文件是否对应当前 checkpoint。
- ONNXRuntime 输出是否在容差内。
- 导出后的模型是否能在不创建 gym 环境的情况下独立运行。
- report 是否记录了输入输出名，而不是只写“导出成功”。

## Spinning Up 对照

Spinning Up 不覆盖现代导出部署。本章是本仓库的工程扩展。学完一个算法不算结束，只有当 inference network 能导出、能被 ONNXRuntime 加载、能通过一致性测试时，训练模型才具备交付意义。

这也是本仓库区别于纯教材的地方：每章不仅要理解公式，还要把模型变成可检查、可保存、可运行的 artifact。

## 学完标准

- 能解释 TorchScript、`torch.export`、ONNX、ONNXRuntime 的区别。
- 能为 DQN、VPG/PPO policy、DDPG/TD3/SAC actor、critic 描述输入输出。
- 能跑通 export demo。
- 能完成一次 PyTorch vs ONNXRuntime 一致性检查。
- 能在 report 中记录导出路径、shape 和误差。
- 能判断一个导出图是否只包含 inference path。
