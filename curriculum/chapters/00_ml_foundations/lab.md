# 00 实验说明书

## 实验目标

你要运行三个监督学习 demo，并能解释每个输出数字和每个 artifact 的用途。不要只追求“命令没报错”，要看 loss、accuracy、导出检查和保存文件是否合理。

## 运行前检查

在仓库根目录运行命令。推荐环境：

```bash
conda activate drl-lab
python -m pip install -e ".[dev]"
```

如果你不用激活环境，也可以在每条命令前使用 `conda run -n drl-lab`。

## 实验 1：线性回归

命令：

```bash
conda run -n drl-lab python curriculum/chapters/00_ml_foundations/code/linear_regression.py
```

期望输出：

```text
final_loss=0.002000
```

数值可能有很小差异。正常范围可以先按 `final_loss < 0.02` 判断。

逐项解释：

- `final_loss` 是最终 MSE loss。
- 它小，说明模型学到的线性关系接近代码里生成标签用的真实规则。
- 它不是 accuracy，也不是百分比。
- 如果 loss 从一开始就很大但最后明显下降，说明训练循环在工作。

artifact：

- `experiments/runs/linear_regression/metrics.csv`：每个 epoch 的 loss。
- `experiments/runs/linear_regression/model.pt`：PyTorch checkpoint，可恢复参数和 optimizer 状态。
- `experiments/runs/linear_regression/model.onnx`：导出的推理模型。
- `experiments/runs/linear_regression/config.json` 或快照文件：记录实验配置，具体文件名由公共工具决定。

异常情况：

- 如果 `final_loss` 大于 `0.1`，先查学习率、epochs、seed 是否被改过。
- 如果报 ONNX 相关错误，先确认环境里有 `onnx` 和 `onnxruntime`。
- 如果 shape 报错，检查输入是否仍是 `[batch, input_dim]`。

## 实验 2：二分类

命令：

```bash
conda run -n drl-lab python curriculum/chapters/00_ml_foundations/code/binary_classifier.py
```

期望输出：

```text
final_accuracy=1.0000
```

正常范围可以先按 `final_accuracy >= 0.95` 判断。这个数据集很简单，通常能接近 1。

逐项解释：

- `final_accuracy` 是训练集准确率。
- `1.0000` 表示训练样本全部分对，不表示所有未来数据都能分对。
- 模型输出叫 logits，不是概率；代码用 sigmoid 和 0.5 阈值算 accuracy。

artifact：

- `experiments/runs/binary_classifier/metrics.csv`：包含 `epoch`、`loss`、`accuracy`。
- `model.pt`：保存训练好的 MLP。
- `model.onnx`：保存导出后的推理图。

异常情况：

- 如果 accuracy 接近 0.5，像随机猜，先查 label 是否仍是 0 和 1。
- 如果 loss 不降，先查是否调用了 `optimizer.step()`。
- 如果你手动加了 sigmoid 再传给 `BCEWithLogitsLoss`，训练会变差，因为这个 loss 已经内部处理 logits。

## 实验 3：图像分类

命令：

```bash
conda run -n drl-lab python curriculum/chapters/00_ml_foundations/code/image_classifier.py
```

期望输出：

```text
final_accuracy=1.0000
```

正常范围可以先按 `final_accuracy >= 0.80` 判断。因为数据是合成条纹图，通常会更高。

逐项解释：

- `final_accuracy` 是验证集准确率，不是训练集准确率。
- 代码每个 epoch 记录 `train_loss`、`val_loss`、`val_accuracy`。
- `train_loss` 下降说明模型更会拟合训练数据。
- `val_accuracy` 高说明模型对没参与训练的合成图像也能分类。

artifact：

- `experiments/runs/image_classifier/metrics.csv`：最值得打开观察的文件。
- `model.pt`：包含 CNN 参数。
- `model.onnx`：导出的 CNN 推理模型。

异常情况：

- 如果报 `expected [batch, 1, 28, 28]`，说明图像 tensor 维度错了。
- 如果报 dtype，检查 labels 是否是 `torch.long`。
- 如果训练很慢，确认 device 仍是 CPU 小模型，不要把样本数改得太大。
- 如果 `val_accuracy` 低但 `train_loss` 低，可能是 overfit 或验证集太少。

## 打开 metrics.csv

示例命令：

```bash
python - <<'PY'
from pathlib import Path
for path in Path("experiments/runs").glob("*/metrics.csv"):
    print(path)
    print(path.read_text().splitlines()[:5])
PY
```

期望输出：每个 CSV 至少有表头和几行数值。

逐项解释：

- 表头告诉你记录了哪些指标。
- 早期 epoch 的 loss 通常比后期大。
- 图像分类的验证指标只在 epoch 结束后记录一次。

## 本章完成标准

你完成本章，不是因为三个脚本都打印了数字，而是因为你能解释：

1. `final_loss` 和 `final_accuracy` 分别是什么。
2. 为什么训练循环里必须先 `zero_grad`，再 `backward`，再 `step`。
3. `metrics.csv`、`model.pt`、`model.onnx` 各自解决什么问题。
4. 哪些异常情况说明模型没有正常学习。

## Ablation

这些 ablation 用于理解机制，不要提交临时改动。

| 改动 | 预期现象 | 解释 |
| --- | --- | --- |
| `linear_regression.py` 的 `epochs=1` | `final_loss` 明显变大 | 参数更新次数太少 |
| `linear_regression.py` 的 `lr=5.0` | loss 可能发散或 ONNX 前就失败 | 学习率过大导致越过低 loss 区域 |
| `binary_classifier.py` 中 accuracy 阈值改成 `0.9` | accuracy 可能下降 | 指标阈值不是训练目标 |
| `image_classifier.py` 的 `samples_per_class` 改小 | 验证指标更不稳定 | 数据少时估计方差更大 |

## 诊断决策树

```text
脚本报错？
  先看 shape/dtype/device 报错位置
脚本跑通但 loss 不降？
  查 optimizer.step、learning rate、target shape
loss 降但 accuracy 差？
  查 loss 与指标是否匹配、阈值是否合理
训练指标好但验证差？
  查 overfit、数据切分、样本数量
导出失败或 passed=False？
  查 ONNX 依赖、example_input shape、模型是否 eval
```

## 记录表

| Demo | 初始/早期 loss | 最终指标 | artifact 是否存在 | 异常解释 |
| --- | --- | --- | --- | --- |
| linear regression | | | | |
| binary classifier | | | | |
| image classifier | | | | |
