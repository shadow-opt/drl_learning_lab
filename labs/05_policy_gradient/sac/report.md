# SAC Report

## Objective

Implement and test the SAC core update equations before adding a full training
loop.

## Experiment Setup

- Environment:
- Seeds:
- Actor hidden sizes:
- Critic hidden sizes:
- Alpha:
- Target entropy:

## Metrics

- Critic loss:
- Actor loss:
- Temperature loss:
- Eval return:
- ONNX max absolute difference:

## Findings

- 

## Debug Notes

- Check sampled actions stay inside action bounds.
- Check log probabilities are finite after the tanh correction.
- Check target Q shape is `[batch]`.
- Check `log_alpha` receives gradients while sampled log probabilities are
  detached in the temperature loss.
