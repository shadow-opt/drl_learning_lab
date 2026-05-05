# Experiment Engineering Notes

Experiments should be reproducible, inspectable, and easy to fail safely.

## Required Habits

- save config snapshots
- save metrics as CSV or JSONL
- save checkpoints
- separate train and eval
- record seed, device, dtype, and dependency versions
- write reports for failed runs
- keep generated artifacts under `experiments/runs/`
