# Repository Guidelines

## Project Structure & Module Organization

`curriculum/chapters/` is the Chinese learning spine. Each chapter contains `lesson.md`, `walkthrough.md`, `lab.md`, `exercises.md`, `hints.md`, `solutions.md`, `report.md`, and `code/`. `src/drl_lab/` contains reusable PyTorch implementations. `tests/` holds unit, smoke, export, and repository-structure checks. `docs/` stores templates, roadmap, debug/export checklists, and engineering guidance. `external/spinningup/` preserves upstream source and MIT license notes.

## Build, Test, and Development Commands

Use the conda environment:

```bash
conda env create -f environment.yml
conda activate drl-lab
python -m pip install -e ".[dev]"
```

Run quality gates before committing:

```bash
conda run -n drl-lab python -m pytest
conda run -n drl-lab ruff check .
conda run -n drl-lab mypy src
```

Run chapter demos from `curriculum/chapters/<chapter>/code/`, for example:

```bash
conda run -n drl-lab python curriculum/chapters/03_tabular_rl/code/q_learning_demo.py
```

## Coding Style & Naming Conventions

Python targets 3.10, uses 4-space indentation, Ruff with 100-character lines, and strict mypy for `src`. Prefer explicit shape comments, deterministic seeds, device/dtype handling, checkpoints, eval paths, and ONNX consistency checks for trainable models.

## Testing Guidelines

Tests use pytest and live in `tests/test_*.py`. New algorithms need focused tests for buffers, losses, update rules, deterministic behavior where practical, and export parity when an inference network exists.

## Commit & Pull Request Guidelines

Use concise imperative commit messages, matching the existing style: `Add DQN export test`, `Document TRPO coverage`. PRs should describe the learning goal, changed chapters/modules, commands run, and any remaining gaps.

## Agent-Specific Instructions

Do not add scripts without curriculum/chapters/lab documentation. Do not create empty placeholder modules. Keep this repository focused on DRL foundations, PyTorch engineering, experiment practice, and export/deployment. Do not expand into Isaac Gym, `legged_gym`, robotics control, sim2real, or large distributed training.
