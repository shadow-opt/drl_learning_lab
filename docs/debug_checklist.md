# Debug Checklist

## Before Training

- Is the global seed set?
- Are environment seeds and action-space seeds set where supported?
- Are observation and action shapes printed or asserted?
- Are tensor dtypes correct?
- Are model, input tensors, and targets on the same device?
- Does the optimizer include all trainable parameters?
- Is the reward scale reasonable?
- Are `terminated` and `truncated` handled intentionally?

## During Training

- Are losses finite?
- Are gradients finite?
- Is gradient norm within a plausible range?
- Is the replay buffer filling with diverse samples?
- Is the target network updated at the intended interval?
- Is exploration decaying too quickly?
- Is entropy collapsing too early in policy-gradient methods?
- Are value losses dominating policy losses?

## After Training

- Does eval disable exploration?
- Does eval use `model.eval()` and `torch.inference_mode()`?
- Can the checkpoint be reloaded?
- Does the same checkpoint produce stable eval results?
- Does ONNXRuntime match PyTorch within tolerance?
- Is the failed or successful experiment written up?
