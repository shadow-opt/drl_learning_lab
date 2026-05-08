# TRPO

## 学习目标

TRPO，Trust Region Policy Optimization，在本仓库中定位为 trust-region 数学阅读模块，而不是完整 benchmark trainer。学完本章后，你应该能解释 surrogate objective、KL constraint、natural gradient、Fisher-vector product、conjugate gradient、step scaling，以及它们如何启发 PPO。

本章不要求你把完整 TRPO 训练循环写出来。目标是理解一个核心问题：policy 更新不能太远，否则用旧 policy 采样的数据就无法可靠评估新 policy。

## 为什么重要

PPO 的 clipping 看起来像经验技巧，但它背后有明确动机：限制新旧 policy 的距离。TRPO 是更显式、更严格的版本。它直接要求平均 KL divergence 不超过阈值。

如果你只学 PPO，不学 TRPO，很容易把 clipping 当作孤立公式；读过 TRPO 后，你会知道 PPO 是用更简单的工程方式近似 trust region。

## 核心直觉

Policy gradient 想朝提高 return 的方向走，但 step 太大可能让 policy 完全变样。TRPO 的直觉是：可以更新，但必须留在旧 policy 附近的“可信区域”内。

这个“附近”用 KL divergence 衡量。KL 小，说明新旧 action distribution 接近；KL 大，说明 policy 已经大幅改变。TRPO 使用二阶近似估计 KL 曲面，然后用 conjugate gradient 找到满足约束的更新方向。

## 数学与公式

TRPO 优化的 surrogate objective：

$$
\max_\theta
\mathbb{E}\left[
\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}\hat{A}
\right]
$$

约束：

$$
\mathbb{E}\left[
D_{KL}(\pi_{\theta_{old}}(\cdot|s)\|\pi_\theta(\cdot|s))
\right] \leq \delta
$$

在局部二阶近似下，KL 可以写成：

$$
\frac{1}{2}x^T H x
$$

`H` 是和 Fisher information 相关的矩阵。Natural gradient 方向需要解线性系统：

$$
Hx = g
$$

其中 `g` 是 policy gradient。实际不显式构造大矩阵 `H`，而是用 Hessian-vector product，并用 conjugate gradient 近似求解。

本仓库在 `src/drl_lab/algorithms/trpo/math.py` 实现两个核心工具：

- `conjugate_gradient`
- `scale_step_to_kl`

`scale_step_to_kl` 的作用是把候选 step 缩放到 KL 限制内。

## 算法流程

完整 TRPO 大致流程：

1. 用旧 policy 采样 trajectories。
2. 计算 advantage。
3. 估计 surrogate objective 的梯度 `g`。
4. 构造 KL 的 Hessian-vector product。
5. 用 conjugate gradient 近似解 `Hx=g`。
6. 根据 KL 上限缩放 step。
7. 使用 line search 检查 surrogate 是否改善且 KL 是否满足约束。
8. 接受或拒绝 policy update。

本仓库当前只实现第 5、6 步的数学工具。这是有意设计：完整 TRPO trainer 工程复杂、脆弱，且不是进入 PPO/DDPG/SAC 的必要前置。

## 代码双索引

- Demo：`curriculum/chapters/05_policy_gradient/trpo/code/trpo_math_demo.py`
- 数学工具：`src/drl_lab/algorithms/trpo/math.py`
- 测试：`tests/test_trpo.py`
- PPO 对照：`curriculum/chapters/05_policy_gradient/ppo/lesson.md`
- Spinning Up mapping：`curriculum/chapters/06_spinningup_track/trpo/spinningup_mapping.md`

阅读顺序：先读本章，再跑 demo，然后读 `math.py` 和测试。最后回到 PPO 章节，解释 clipping 如何替代显式 KL 约束。

## 实验任务

```bash
conda run -n drl-lab python curriculum/chapters/05_policy_gradient/trpo/code/trpo_math_demo.py
```

观察：

- conjugate gradient residual 是否下降；
- KL limit 变小时 step scale 是否变小；
- step scaling 是否满足目标 KL；
- 测试如何覆盖这些数学工具。

在 `curriculum/chapters/05_policy_gradient/trpo/report.md` 中写出 TRPO 和 PPO 的关系。

## Debug Checklist

- KL 方向是否写成 `old || new`。
- conjugate gradient 输入是否代表线性算子 `H*v`。
- residual 是否逐步下降。
- step scaling 是否处理零向量或极小 curvature。
- 是否误以为本仓库已有完整 TRPO trainer。
- 是否能解释为什么 line search 在完整 TRPO 中必要。

## Spinning Up 对照

Spinning Up TRPO 是本章主参考。它完整讲解 trust region、surrogate objective、conjugate gradient 和 line search。本仓库只吸收其中最关键的数学部件，并将其隔离成可测试函数，避免在初学阶段维护一个复杂且脆弱的完整实现。

学习 TRPO 时应重点问：PPO clipping 和 TRPO KL constraint 都在限制什么？答案是同一个：限制 policy 更新幅度，让旧数据对新 policy 仍有参考价值。

## 学完标准

- 能写出 TRPO surrogate objective 和 KL constraint。
- 能解释 natural gradient 为什么涉及 `Hx=g`。
- 能说明 conjugate gradient 在这里解决什么问题。
- 能跑通 TRPO math demo。
- 能解释 PPO 和 TRPO 的关系。
- 能明确本仓库当前 TRPO 是数学阅读模块，不是完整 trainer。
