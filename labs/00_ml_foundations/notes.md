# ML Foundations Notes

Focus on the minimum machine learning background needed before deep RL.

## Core Ideas

- A model maps inputs to predictions.
- A loss function turns prediction error into a scalar objective.
- Gradient descent updates parameters to reduce the loss.
- Generalization matters more than memorizing the training set.
- A baseline should be simple enough to debug.

## Required Topics

- supervised learning
- train/validation/test split
- linear regression
- logistic regression
- classification metrics
- overfitting and regularization
- normalization
- learning curves

## Engineering Habits

- Write down input and output shapes.
- Start with a tiny dataset that the model can overfit.
- Track loss before adding complexity.
- Save the config and seed for each experiment.
