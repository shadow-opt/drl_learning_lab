# 00 代码逐行走读

本章逐行走读三个 demo：`linear_regression.py`、`binary_classifier.py`、`image_classifier.py`。导入语句会合并讲，因为它们主要是在准备工具；影响理解的变量、函数、训练循环、输出和 artifact 都会解释。

## 读代码前先定目标

不要急着背 API。你要在每个文件里找同一条主线：

```text
config -> seed -> dataset -> model -> loss -> optimizer -> train loop -> metrics -> checkpoint -> export -> check -> print
```

这条主线以后会在 DRL 里反复出现，只是 dataset 会换成 replay buffer 或 rollout，loss 会换成 Bellman loss 或 policy loss。

## `linear_regression.py` 逐行走读

`from __future__ import annotations` 让类型注解更现代，学习重点不在这里。

`dataclass` 用来写配置类，`Path` 用来表示输出目录。`torch` 是张量库，`nn` 是神经网络模块，`Adam` 是 optimizer。

`CheckpointMetadata`、`save_checkpoint`、`save_run_snapshots`、`export_to_onnx`、`CsvLogger`、`compare_pytorch_onnx`、`set_global_seed` 是本仓库公共工具。它们让 demo 不只是训练，还会保存配置、指标、模型和导出检查。

`@dataclass(frozen=True)` 定义不可变配置。学习点是：实验参数集中写，方便复现实验。

`LinearRegressionConfig.seed = 42` 控制随机数。学习点是：同一个 seed 下，随机初始化和随机数据更容易复现。

`n_samples = 256` 表示生成 256 条训练样本。

`input_dim = 3` 表示每条输入有 3 个特征，所以输入 shape 会是 `[256, 3]`。

`lr = 0.05` 是学习率，控制 optimizer 每次改参数的步子大小。

`epochs = 200` 表示完整训练循环跑 200 次。

`run_dir = Path("experiments/runs/linear_regression")` 是 artifact 输出目录。

`class LinearRegressor(nn.Module)` 定义一个 PyTorch 模型。学习点是：模型类继承 `nn.Module`，可训练层放在里面。

docstring 里的 `Input shape: [batch, input_dim]` 告诉你输入必须是二维。第一维是样本数，第二维是特征数。

`super().__init__()` 初始化父类。没有它，PyTorch 不能正确登记子模块。

`self.linear = nn.Linear(input_dim, 1)` 创建线性层。它把每个 3 维输入变成 1 个预测值。

`def forward(self, x)` 定义模型被调用时发生什么。`model(x)` 实际会进入 `forward`。

`if x.ndim != 2` 是 shape 防线。学习点是：模型越早检查 shape，错误越容易定位。

`output = self.linear(x)` 做线性变换，输出 shape 是 `[batch, 1]`。

`return output` 返回预测。

`make_dataset` 生成离线数据。它返回 `(x, y)` 两个 tensor。

`x = torch.randn(config.n_samples, config.input_dim)` 生成随机输入，shape 是 `[256, 3]`。

`true_w = torch.tensor([[2.0], [-3.0], [0.5]])` 是真实权重，shape 是 `[3, 1]`。

`true_b = torch.tensor([0.25])` 是真实 bias。

`y = x @ true_w + true_b + 0.05 * torch.randn(...)` 生成标签。`x @ true_w` 得到 `[256, 1]`，后面的噪声模拟真实数据不完美。

`train(config)` 是主训练函数。它返回训练好的模型和最终 loss。

`set_global_seed(config.seed)` 固定随机性。学习点是：先固定 seed，再生成数据和模型。

`config.run_dir.mkdir(...)` 确保输出目录存在。

`save_run_snapshots(config, config.run_dir)` 保存实验配置快照。以后看到结果，可以知道这次实验用的参数。

`x, y = make_dataset(config)` 拿到训练数据。

`model = LinearRegressor(config.input_dim)` 创建模型，此时参数还没有学好。

`optimizer = Adam(model.parameters(), lr=config.lr)` 告诉 optimizer 要更新哪些参数。

`loss_fn = nn.MSELoss()` 定义回归 loss。

`metrics_path = config.run_dir / "metrics.csv"` 指标会写到 CSV。

`with CsvLogger(...) as logger` 打开日志器，字段是 `epoch` 和 `loss`。

`for epoch in range(config.epochs)` 开始训练 200 轮。

`model.train()` 切到训练模式。线性层本身不受影响，但这是训练代码的标准习惯。

`pred = model(x)` 前向预测。

`loss = loss_fn(pred, y)` 把预测和标签的差距变成一个标量。

`optimizer.zero_grad(set_to_none=True)` 清掉旧梯度。

`loss.backward()` 计算当前 loss 对参数的 gradient。

`optimizer.step()` 更新参数。这一行之后模型真的变了。

`logger.log(...)` 记录当前 epoch 的 loss。`loss.detach()` 表示只取数值，不把日志记录放进计算图。

循环结束后，`final_loss = float(loss_fn(model(x), y).detach())` 用最终模型再算一次 loss。

`save_checkpoint(...)` 保存模型参数、optimizer 状态、step、seed 和 `final_loss`。

`example_input = x[:4]` 取 4 条样本作为导出示例输入。学习点是：导出需要知道输入长什么样。

`onnx_path = export_to_onnx(...)` 把 PyTorch 模型导出为 ONNX。

`result = compare_pytorch_onnx(...)` 用 ONNXRuntime 跑同一个输入，对比 PyTorch 输出。

`if not result.passed` 表示导出结果和 PyTorch 差太多就失败。

`return model, final_loss` 给测试或其他代码复用。

`main()` 调用训练并打印 `final_loss=...`。你运行脚本时看到的就是这一行。

## `binary_classifier.py` 逐行走读

这个文件和线性回归主线一样，但任务从“预测连续数字”变成“判断 0 类还是 1 类”。

`BinaryClassifierConfig.seed = 123` 换了 seed，避免所有实验都完全一样。

`n_per_class = 128` 表示每个类别生成 128 个点，总样本数是 256。

`lr = 0.03`、`epochs = 200` 是训练超参数。

`make_dataset` 生成二维平面上的两团点。

`class_0 = torch.randn(...) * 0.4 + torch.tensor([-1.0, -1.0])` 生成围绕 `(-1, -1)` 的点。

`class_1 = ... + torch.tensor([1.0, 1.0])` 生成围绕 `(1, 1)` 的点。

`x = torch.cat([class_0, class_1], dim=0)` 把两类样本接起来，shape 是 `[256, 2]`。

`y = torch.cat([zeros, ones], dim=0)` 生成标签，shape 是 `[256, 1]`。0 表示第 0 类，1 表示第 1 类。

`accuracy_from_logits` 把模型输出转成准确率。

`torch.sigmoid(logits)` 把 logits 压到 0 到 1。学习点是：二分类模型通常输出 logits，不直接输出概率。

`>= 0.5` 用阈值得到类别预测。

`predictions == labels` 得到每个样本是否预测正确。

`model = MLP(input_dim=2, hidden_sizes=[16, 16], output_dim=1)` 使用公共 MLP。输入是二维点，输出是一个 logit。

`loss_fn = nn.BCEWithLogitsLoss()` 是二分类常用 loss。它内部包含 sigmoid 的数值稳定版本，所以训练时不要自己先 sigmoid。

训练循环里的 `logits = model(x)`、`loss = loss_fn(logits, y)`、`zero_grad`、`backward`、`step` 和回归完全同构。

`logger.log` 多记录了 `accuracy`。学习点是：loss 是训练优化目标，accuracy 是人更容易读懂的任务指标。

训练后 `model.eval()` 切到推理模式。

`with torch.inference_mode()` 表示只做推理，不需要梯度，速度更快、内存更省。

`final_accuracy = accuracy_from_logits(model(x), y)` 计算最终训练集准确率。

checkpoint 里的 `extra={"final_accuracy": final_accuracy}` 保存最终准确率。

ONNX 导出和一致性检查与线性回归一样。

`main()` 打印 `final_accuracy=...`。

常见误解：`final_accuracy=1.0000` 不代表模型懂所有分类问题。它只说明这个简单二维数据集被分开了。

## `image_classifier.py` 逐行走读

这个文件更接近真实训练：有 Dataset、DataLoader、batch、train split、val split、CNN、训练集 loss 和验证集 accuracy。

`ImageClassifierConfig.num_classes = 10` 表示 10 类图像。

`samples_per_class = 24` 表示每类 24 张，总共 240 张合成图像。

`batch_size = 32` 表示每次训练最多看 32 张。

`epochs = 8` 比前两个 demo 少，因为每个 epoch 已经包含多个 batch。

`device = "cpu"` 默认在 CPU 上跑，保证没有 GPU 也能完成。

`PatternImageDataset(Dataset[...])` 定义一个离线图像数据集。学习点是：Dataset 负责“按 index 返回一个样本”。

`__init__` 先检查类别数和每类样本数是否合法。

`generator = torch.Generator().manual_seed(seed)` 为数据噪声单独固定随机数。

外层 `for label in range(num_classes)` 为每个类别生成基础图案。

`base = self._base_pattern(label)` 生成这个类别固定的条纹模板。

内层循环为同一类别生成多张图。`noise = 0.03 * torch.randn(...)` 添加小噪声。

`image = (base + noise).clamp(0.0, 1.0)` 保证像素值留在 0 到 1。

`self.images = torch.stack(images).to(dtype=torch.float32)` 把图片列表叠成一个 tensor，shape 是 `[240, 1, 28, 28]`。

`self.labels = torch.tensor(labels, dtype=torch.long)` 分类标签必须是 `long`，因为 `CrossEntropyLoss` 要求类别编号。

`__len__` 返回样本数。

`__getitem__` 返回一张图和一个标签。

`_base_pattern` 里 `image = torch.zeros(1, 28, 28)` 创建单通道 28x28 图像。

`row`、`col` 根据 label 改变条纹位置，让不同类别有不同模式。

三行切片赋值是在图像上画横条、竖条和亮块。

`SmallImageClassifier(nn.Module)` 是小型 CNN。

`self.features` 包含卷积、ReLU、池化。学习点是：卷积提取局部图案，池化缩小空间尺寸。

第一次 `MaxPool2d(2)` 把 28x28 变 14x14，第二次变 7x7。

`self.classifier` 先 `Flatten`，再用线性层输出 `num_classes` 个 logits。

`nn.Linear(16 * 7 * 7, 64)` 里的 `16 * 7 * 7` 来自卷积输出 shape。

`forward` 检查 `images.ndim != 4`。图像 batch 必须是 `[batch, channel, height, width]`。

`features = self.features(images)` 提取特征。

`logits = self.classifier(features)` 得到每类分数，shape 是 `[batch, 10]`。

`accuracy_from_logits` 用 `argmax(dim=-1)` 取分数最高的类别。

`ImageLoader = DataLoader[...]` 是类型别名，让函数签名更清楚。

`make_loaders` 创建数据集，并按 80% 训练、20% 验证切分。

`random_split(..., generator=generator)` 用固定 seed 切分，避免每次训练集都不同。

`DataLoader(train_dataset, batch_size=..., shuffle=True)` 训练集打乱顺序。

`DataLoader(val_dataset, ..., shuffle=False)` 验证集不需要打乱。

`evaluate` 是验证函数。它先 `model.eval()`，再用 `torch.inference_mode()` 禁用梯度。

`images.to(device=device, dtype=torch.float32)` 确保图像在正确设备和 dtype。

`labels.to(device=device, dtype=torch.long)` 确保标签 dtype 符合 loss 要求。

`loss = loss_fn(logits, labels)` 计算多分类 loss。

`total_loss += loss * 样本数` 是为了最后得到按样本平均的 loss。

`total_correct` 和 `total_examples` 用来算准确率。

`train` 函数里先固定 seed，再解析 device，再保存快照，再创建 loader。

`model = SmallImageClassifier(...).to(device=device)` 把模型参数放到同一设备。

训练循环外层按 epoch，内层按 batch。

每个 batch 都重复：搬数据到 device，前向得到 logits，算 loss，清梯度，反向传播，optimizer 更新。

`running_loss += float(loss.detach()) * int(labels.numel())` 累加训练 loss。

每个 epoch 结束后 `val_loss, val_accuracy = evaluate(...)`，用验证集检查泛化。

CSV 记录 `epoch`、`train_loss`、`val_loss`、`val_accuracy`。

训练结束后保存 checkpoint，然后从验证 loader 取 4 张图作为 ONNX 示例输入。

`compare_pytorch_onnx` 通过才说明导出的推理模型和 PyTorch 模型在示例输入上足够一致。

`main()` 打印 `final_accuracy=...`。

常见误解：验证 accuracy 低时，不要只盯着代码是否报错。要检查数据、label dtype、模型输入 shape、学习率、训练轮数和 train/eval 是否切换正确。

## 公式对应

把三个 demo 都映射到同一个优化公式：

```text
theta_{t+1} = theta_t - optimizer_step(gradient_theta loss)
```

`linear_regression.py`：

- `model(x)` 对应 `f_theta(X)`。
- `nn.MSELoss()` 对应 `(1/N) sum_i (y_hat_i - y_i)^2`。
- `loss.backward()` 对应计算 `gradient_theta R_hat(theta)`。
- `optimizer.step()` 对应更新 `theta`。

`binary_classifier.py`：

- `logits = model(x)` 对应未归一化分数 `z`。
- `BCEWithLogitsLoss(logits, y)` 对应二分类负对数似然。
- `accuracy_from_logits` 是指标，不进入训练目标。

`image_classifier.py`：

- `logits = model(images)` 的 shape 是 `[batch, 10]`。
- `CrossEntropyLoss(logits, labels)` 对应多分类负对数似然。
- `evaluate` 对应估计验证风险，不更新参数。

## shape 对照表

| 文件 | 输入 shape | 输出 shape | target shape | loss |
| --- | --- | --- | --- | --- |
| `linear_regression.py` | `[256, 3]` | `[256, 1]` | `[256, 1]` | MSE |
| `binary_classifier.py` | `[256, 2]` | `[256, 1]` | `[256, 1]` | BCE logits |
| `image_classifier.py` | `[batch, 1, 28, 28]` | `[batch, 10]` | `[batch]` | CrossEntropy |

公开课级读代码要求：每看到一个 loss，就能说明 pred 和 target 的 shape 为什么合法。
