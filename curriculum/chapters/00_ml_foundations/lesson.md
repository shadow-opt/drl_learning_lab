# 00 机器学习基础课

## 这章为什么存在

后面的 DRL 看起来会有很多新词：Q 网络、policy、actor、critic、target network、replay buffer。它们最后都会落回同一件事：用数据算出一个 loss，让 PyTorch 根据 gradient 更新一批 parameter。

本章不假设你会机器学习。你只需要会 Python 函数、列表、循环和一点点矩阵乘法直觉。学完以后，你应该能看懂一个训练循环为什么会让 loss 下降，也能分辨“模型在学习”和“代码只是跑完了”不是同一件事。

## 先抓住一句话

机器学习训练就是：model 先根据输入猜答案，loss 把“猜错多少”变成一个数字，gradient 说明每个参数该往哪个方向改，optimizer 真正改参数。batch 是一次拿来训练的一小组样本，overfit 是模型记住训练数据但不能泛化到新数据，generalization 是模型对没见过的数据也能表现好。

## model：模型就是带参数的函数

白话解释：普通 Python 函数的规则通常是你写死的；机器学习 model 的规则有一部分是“可训练数字”。这些数字叫 parameter。训练前参数通常是随机的，所以一开始预测会错。

最小例子：

```python
def hand_written_rule(x: float) -> float:
    return 2.0 * x + 0.25
```

线性模型做的是类似的事，但 `2.0` 和 `0.25` 不是你提前知道的，而是模型要从数据里学出来：

```text
y_hat = x @ weight + bias
```

常见误解：模型不是“自动理解世界”。它只是在当前结构允许的范围内调参数。线性模型只能画直线或超平面，CNN 更适合图像，但也仍然是函数。

和 DRL 的关系：DQN 里的 Q 网络、PPO 里的 policy network、SAC 里的 critic，本质上都是 model。它们输入 state 或 observation，输出 action 分数、动作分布或 value。

## loss：把错变成一个数字

白话解释：人可以看一眼预测说“差得有点多”，电脑需要一个明确数字。loss 越小，通常代表预测越接近目标。

最小例子：如果目标是 `10`，预测是 `8`，误差是 `-2`。MSE 会平方误差：

```text
(8 - 10) ** 2 = 4
```

如果预测变成 `9.5`：

```text
(9.5 - 10) ** 2 = 0.25
```

`0.25` 比 `4` 小，所以第二个预测更好。

常见误解：loss 小不一定代表模型在真实任务上好。模型可能只是在训练集上 loss 小，这就可能 overfit。分类任务也不总看 MSE，二分类常用 `BCEWithLogitsLoss`，多分类常用 `CrossEntropyLoss`。

和 DRL 的关系：DQN 的 loss 衡量 Q 估计和 Bellman target 的差距；policy gradient 的 loss 经常不是“预测标签错多少”，而是让高回报动作更可能出现。形式不同，但训练仍然需要一个可反向传播的标量。

## gradient：告诉参数怎么改

白话解释：gradient 是 loss 对参数的敏感程度。它回答：“如果这个参数稍微变大一点，loss 会怎么变？”如果变大会让 loss 变大，optimizer 通常就让它变小一点。

最小例子：只有一个参数 `w`，loss 是：

```text
loss = (w - 3) ** 2
```

当 `w = 1` 时，loss 是 `4`，gradient 是负数，说明把 `w` 调大能让 loss 降低。当 `w = 5` 时，gradient 是正数，说明把 `w` 调小能让 loss 降低。

常见误解：gradient 不是最终答案，也不是“保证一步到最优”。学习率太大可能越过好位置，学习率太小可能很久都不动。

和 DRL 的关系：DRL 训练不稳定时，常见原因包括 target 噪声大、回报尺度大、gradient 爆炸或更新目标变化太快。先理解 gradient，后面才能理解为什么要裁剪梯度、标准化 advantage、使用 target network。

## optimizer：真正改参数的人

白话解释：`loss.backward()` 只把梯度算出来，参数还没变。`optimizer.step()` 才会读取梯度并更新参数。

最小训练循环：

```python
prediction = model(x)
loss = loss_fn(prediction, y)
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
```

逐项理解：

- `prediction = model(x)`：用当前参数做预测。
- `loss = loss_fn(prediction, y)`：计算当前预测错多少。
- `zero_grad`：清掉上一轮梯度，因为 PyTorch 默认会累加梯度。
- `backward`：沿计算图反向计算每个参数的 gradient。
- `step`：用 optimizer 的规则更新参数。

常见误解：不能省略 `zero_grad`。如果不清梯度，本轮梯度会和上一轮混在一起，更新方向会变得不可控。

和 DRL 的关系：后面所有可训练算法都会出现这个顺序。你能认出这五步，就能在 DQN、VPG、PPO、DDPG、TD3、SAC 里定位“真正学习发生在哪里”。

## batch：一次训练的一小组样本

白话解释：如果每次只看一个样本，更新会很抖；如果一次看完整数据，可能慢且占内存。batch 是折中：每次拿一小组样本估计梯度。

最小例子：图像分类里 `batch_size=32`，意思是每次从 DataLoader 拿 32 张图像和 32 个标签。模型输入 shape 是 `[32, 1, 28, 28]`，输出 shape 是 `[32, 10]`。

常见误解：batch 不是类别数。`[32, 10]` 里的 `32` 是样本数，`10` 是每个样本对应 10 个类别的 logits。

和 DRL 的关系：replay buffer 采样出来的 transition batch、PPO 的 minibatch、SAC 的 batch，都用同一个思想：用一批样本估计一次更新。

## overfit 和 generalization

白话解释：overfit 是“考试前把原题背下来了，但换一道题就不会”。generalization 是“理解了规律，所以新题也能做”。

最小例子：如果训练集 accuracy 是 100%，验证集 accuracy 只有 50%，就要怀疑 overfit。图像 demo 会记录 `train_loss`、`val_loss`、`val_accuracy`，验证集就是用来观察泛化的。

常见误解：训练 loss 一直下降不等于模型越来越有用。你还要看验证指标、测试环境表现，或者至少看没参与训练的数据。

和 DRL 的关系：智能体可能只会某个随机种子、某张地图或某个初始状态。后面做 eval path、固定 seed、保存 metrics，就是为了区分“训练跑通”和“策略真的可靠”。

## 三个 demo 分别学什么

`linear_regression.py` 学最小监督学习闭环：数据、线性模型、MSE、Adam、checkpoint、ONNX 检查。

`binary_classifier.py` 学分类和 logits：模型输出不是概率，先经过 sigmoid 才能按 0.5 得到类别。

`image_classifier.py` 学 batch、DataLoader、train/eval 分离和验证集：图像输入多了 channel、高度、宽度，训练指标也多了验证 loss 和验证 accuracy。

## 学完检查

你应该能用自己的话回答：

1. 为什么模型一开始会错。
2. loss 为什么能表示错得多严重。
3. gradient 和 optimizer 分别负责什么。
4. 一轮训练循环每一步做什么。
5. batch 的第 0 维为什么通常叫 batch 维。
6. overfit 和 generalization 的区别是什么。

## 公开课扩展：形式化定义

监督学习的最小形式可以写成一个四元组：

```text
D = {(x_i, y_i)}_{i=1}^N
f_theta: x -> y_hat
L(y_hat, y) -> scalar
theta <- optimizer(theta, gradient)
```

`D` 是数据集，`x_i` 是第 `i` 个输入，`y_i` 是目标，`theta` 是所有可训练参数。模型 `f_theta` 输入 `x_i` 得到预测 `y_hat_i`，loss 把预测和目标比较成一个标量。训练不是直接“让模型聪明”，而是在最小化经验风险：

```text
R_hat(theta) = (1 / N) * sum_i L(f_theta(x_i), y_i)
```

经验风险是训练集上的平均 loss。真实目标通常是新样本上的风险，但新样本还没出现，所以训练只能用经验风险作为代理。

## 公开课扩展：线性代数视角

线性回归 demo 中，`x` 的 shape 是 `[N, 3]`，`true_w` 的 shape 是 `[3, 1]`。矩阵乘法：

```text
X @ w -> [N, 1]
```

这里每一行是一个样本，每一列是一个 feature。线性模型实际学习：

```text
y_hat = Xw + b
```

shape 约束是严肃的：`[N, 3] @ [3, 1]` 合法，`[N, 3] @ [1, 3]` 不合法。后面 DRL 的 Q 网络同理，`[batch, obs_dim]` 输入必须和第一层线性层的 `input_dim` 对上。

## 公开课扩展：MSE 推导

单样本 MSE 可以写成：

```text
L(w) = (w x - y)^2
```

对 `w` 求导：

```text
dL/dw = 2 * (w x - y) * x
```

如果 `w x` 大于 `y`，导数通常为正，梯度下降会让 `w` 变小；如果 `w x` 小于 `y`，导数可能为负，梯度下降会让 `w` 变大。更新式是：

```text
w_new = w_old - lr * dL/dw
```

例题：`x=2, y=10, w=1`。

```text
y_hat = 1 * 2 = 2
L = (2 - 10)^2 = 64
dL/dw = 2 * (2 - 10) * 2 = -32
如果 lr=0.1，w_new = 1 - 0.1 * (-32) = 4.2
```

这一步很大，说明学习率会显著影响训练稳定性。真实训练用 batch 平均梯度和 Adam，所以不会完全等同这个手算，但方向直觉一致。

## 公开课扩展：BCE 与 CrossEntropy

二分类 demo 使用 logits。logit 是还没压到 0 到 1 的原始分数。sigmoid 把 logit 转成概率样数值：

```text
p = 1 / (1 + exp(-z))
```

BCE 的目标是：标签为 1 时让 `p` 接近 1，标签为 0 时让 `p` 接近 0。`BCEWithLogitsLoss` 把 sigmoid 和 BCE 合在一起，数值上更稳定。

多分类图像 demo 使用 CrossEntropy。模型输出 `[batch, num_classes]` logits，标签是 `[batch]` 的整数类别。CrossEntropy 等价于先做 softmax，再取真实类别概率的负对数。真实类别概率越高，loss 越小。

常见误解：二分类的 sigmoid 和多分类的 softmax 都能产生概率样数值，但它们的 shape 和语义不同。二分类 demo 输出 `[batch, 1]`，图像分类 demo 输出 `[batch, 10]`。

## 公开课扩展：优化与泛化

梯度下降优化的是训练目标，但课程关心的是泛化。可以把误差来源粗略拆成：

```text
generalization error = bias + variance + noise
```

这不是本章要严格证明的定理，而是诊断框架：

- bias 高：模型太弱或特征不够，训练 loss 也降不下去。
- variance 高：模型记住训练集，验证集表现差。
- noise 高：数据本身有随机扰动，loss 不可能到 0。

线性回归 demo 的标签加了 `0.05 * torch.randn(...)`，所以最终 MSE 很小但不应期待严格等于 0。图像分类 demo 有验证集，所以更适合讨论 generalization。

## 公开课扩展：本章定理化小结

你可以把本章记成三个 invariants：

1. 训练循环 invariant：一次参数更新必须包含 forward、loss、zero grad、backward、step。
2. shape invariant：模型输出和 target 必须能被 loss 合法比较。
3. evidence invariant：判断训练是否正常，要看 metrics 和验证逻辑，不能只看脚本退出码。

后续 DRL 的复杂度来自数据不是固定标签，而是智能体和环境交互产生；但只要出现神经网络训练，这三个 invariant 仍然成立。
