# SAC Exercises

1. Verify that `SquashedGaussianActor.sample` returns bounded actions and
   finite log probabilities for random observations.
2. Remove the `tanh` log-prob correction and describe how the actor loss
   changes.
3. Compare fixed `alpha` against learned `log_alpha` on a tiny replay batch.
4. Add a Pendulum training loop with checkpoint, eval, CSV logging, and ONNX
   export.
5. Write a report comparing DDPG, TD3, and SAC on the same seed set.
