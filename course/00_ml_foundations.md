# 00 ML Foundations

## 学习目标

本章目标不是把机器学习重新学成一门完整课程，而是补齐后面写 Deep RL 时一定会反复用到的监督学习直觉和 PyTorch 训练习惯。学完后你应该能解释 model、loss、gradient、generalization、overfitting、normalization 的关系，能读懂一个最小训练循环，能根据 loss 曲线判断训练是否在正常发生。

工程上，你需要能跑通线性回归、二分类 MLP 和小型图像分类 demo，理解输入输出 shape，知道训练产物里为什么要有 config、metrics、checkpoint 和 ONNX 文件。

## 为什么重要

Deep RL 的困难经常被误判为“算法太难”。实际调试时，很多问题来自更基础的地方：张量 shape 错了、loss 没有下降、数据尺度不合理、训练和 eval 模式混用、随机种子不固定、checkpoint 不能恢复。先用监督学习把这些问题暴露出来，成本远低于直接在 PPO 或 SAC 里 debug。

监督学习还有一个价值：它给你一个简单的优化参照。在线性回归中，模型应该很快拟合；二分类中，decision boundary 应该稳定；图像分类中，batch shape 应该是 `[batch, channels, height, width]`。这些小任务是后面检查 RL 神经网络是否正常的基准。

## 核心直觉

机器学习训练可以先理解成四件事：

1. 模型把输入映射成预测。
2. loss 把预测错误压成一个标量。
3. autograd 计算参数对 loss 的梯度。
4. optimizer 用梯度更新参数。

如果模型太简单，loss 降不下去，叫 underfitting。如果模型能记住训练集但不能泛化，叫 overfitting。训练不是只看 final loss，而是看训练集、验证集、测试集之间的关系。

在 RL 里，数据不是固定数据集，而是 agent 和 environment 交互产生的。但神经网络训练仍然遵循同一套基本规律：输入尺度要稳定，loss 要可解释，梯度要有限，eval 要和 train 分开。

## 数学与公式

线性回归的形式是：

$$
\hat{y} = xW + b
$$

常见 loss 是均方误差：

$$
L(\theta) = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2
$$

梯度下降的参数更新是：

$$
\theta \leftarrow \theta - \alpha \nabla_\theta L(\theta)
$$

这里 `theta` 表示模型参数，`alpha` 是 learning rate。这个公式会在后面不断变形：DQN 的 loss 是 TD error，policy gradient 的目标是 expected return，SAC 还会加入 entropy 项，但“构造标量目标，然后反向传播”这个工程模式不变。

二分类通常输出 logit，再用 binary cross entropy。logit 是还没过 sigmoid 的分数。工程上更推荐直接使用 `BCEWithLogitsLoss`，因为它把 sigmoid 和 BCE 合在一起，数值更稳定。

## 算法流程

一个最小监督学习循环：

1. 固定 seed。
2. 构造 dataset 和 dataloader。
3. 创建 model、optimizer、loss function。
4. 对每个 batch：
   - 将输入和标签移动到目标 device；
   - 前向计算预测；
   - 计算 loss；
   - `optimizer.zero_grad()`；
   - `loss.backward()`；
   - `optimizer.step()`。
5. 切到 eval 模式，使用 `torch.inference_mode()` 评估。
6. 保存 metrics、checkpoint。
7. 导出 ONNX，并比较 PyTorch 与 ONNXRuntime 输出。

## 代码双索引

- 线性回归 demo：`labs/00_ml_foundations/code/linear_regression.py`
- 二分类 demo：`labs/00_ml_foundations/code/binary_classifier.py`
- 小型图像分类 demo：`labs/00_ml_foundations/code/image_classifier.py`
- 测试：`tests/test_supervised_examples.py`
- 通用 checkpoint：`src/drl_lab/common/checkpoint.py`
- 通用导出：`src/drl_lab/common/export.py`

这些 demo 故意不依赖大型数据集。它们的目的不是刷 benchmark，而是训练基本工程动作：shape、seed、metrics、checkpoint、export。

## 实验任务

运行：

```bash
conda run -n drl-lab python labs/00_ml_foundations/code/linear_regression.py
conda run -n drl-lab python labs/00_ml_foundations/code/binary_classifier.py
conda run -n drl-lab python labs/00_ml_foundations/code/image_classifier.py
```

观察每个实验是否写出 `metrics.csv`、checkpoint 和 ONNX 文件。检查 loss 是否下降，检查 eval 指标是否符合任务难度。故意把 learning rate 改大一次，记录 loss 是否震荡或发散。

## Debug Checklist

- 输入 shape 是否和模型 `forward` 注释一致。
- label dtype 是否正确：分类标签、回归标签不要混用。
- 是否忘记 `optimizer.zero_grad()`。
- eval 时是否使用 `model.eval()` 和 `torch.inference_mode()`。
- 训练和验证数据是否意外相同。
- ONNX 输出是否和 PyTorch 输出在容差内一致。

## Spinning Up 对照

Spinning Up 主要讲 DRL，不负责完整 ML 基础。本章是进入 Spinning Up 前的补课层。你不需要在这里读 Spinning Up，但需要带着这些训练循环习惯进入后续 DQN、VPG、PPO 和 SAC。

## 学完标准

- 能解释 loss、gradient、optimizer、overfitting 的关系。
- 能手写一个最小训练循环。
- 能说明 `[batch, feature]` 和 `[batch, channels, height, width]` 的含义。
- 能跑通三个 demo，并在 `labs/00_ml_foundations/report.md` 写一次观察和一次失败记录。
