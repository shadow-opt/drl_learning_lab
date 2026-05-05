from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from drl_lab.algorithms.dqn.agent import DQNAgent
from drl_lab.algorithms.dqn.config import DQNConfig
from drl_lab.common.checkpoint import load_checkpoint
from drl_lab.common.device import resolve_device
from drl_lab.common.export import export_to_onnx
from drl_lab.common.onnx_check import compare_pytorch_onnx


def export_checkpoint(
    checkpoint_path: Path,
    output_path: Path,
    obs_dim: int,
    n_actions: int,
    config: DQNConfig,
) -> None:
    device = resolve_device("auto")
    agent = DQNAgent(obs_dim, n_actions, config, device)
    load_checkpoint(checkpoint_path, agent.q_network, map_location=device)
    example_input = torch.zeros(4, obs_dim, device=device)
    export_to_onnx(agent.q_network, example_input, output_path)
    result = compare_pytorch_onnx(agent.q_network, output_path, example_input)
    if not result.passed:
        raise RuntimeError(f"ONNX consistency failed: {result}")
    print(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a DQN Q-network checkpoint to ONNX.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--obs-dim", type=int, default=4)
    parser.add_argument("--n-actions", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=DQNConfig.hidden_size)
    args = parser.parse_args()

    config = replace(DQNConfig(), hidden_size=args.hidden_size)
    export_checkpoint(args.checkpoint, args.output, args.obs_dim, args.n_actions, config)


if __name__ == "__main__":
    main()
