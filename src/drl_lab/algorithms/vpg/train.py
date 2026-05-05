from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Discrete
from numpy.typing import NDArray

from drl_lab.algorithms.vpg.agent import VPGAgent
from drl_lab.algorithms.vpg.buffer import make_trajectory_batch
from drl_lab.algorithms.vpg.config import VPGConfig
from drl_lab.algorithms.vpg.eval import evaluate
from drl_lab.common.checkpoint import CheckpointMetadata, save_checkpoint
from drl_lab.common.device import resolve_device
from drl_lab.common.experiment import save_run_snapshots
from drl_lab.common.export import export_to_onnx
from drl_lab.common.logging import CsvLogger
from drl_lab.common.onnx_check import compare_pytorch_onnx
from drl_lab.common.seed import set_global_seed


def _append_episode_tensors(
    all_obs: list[torch.Tensor],
    all_actions: list[torch.Tensor],
    all_returns: list[torch.Tensor],
    all_advantages: list[torch.Tensor],
    obs_buffer: list[NDArray[np.float32]],
    action_buffer: list[int],
    reward_buffer: list[float],
    value_buffer: list[float],
    log_prob_buffer: list[float],
    gamma: float,
    device: torch.device,
) -> None:
    obs_tensor = torch.as_tensor(np.asarray(obs_buffer), dtype=torch.float32, device=device)
    action_tensor = torch.as_tensor(action_buffer, dtype=torch.int64, device=device)
    reward_tensor = torch.as_tensor(reward_buffer, dtype=torch.float32, device=device)
    value_tensor = torch.as_tensor(value_buffer, dtype=torch.float32, device=device)
    log_prob_tensor = torch.as_tensor(log_prob_buffer, dtype=torch.float32, device=device)

    batch = make_trajectory_batch(
        obs_tensor,
        action_tensor,
        reward_tensor,
        value_tensor,
        log_prob_tensor,
        gamma=gamma,
        normalize_advantage=False,
    )
    all_obs.append(batch.obs)
    all_actions.append(batch.actions)
    all_returns.append(batch.returns)
    all_advantages.append(batch.advantages)


def train(config: VPGConfig) -> dict[str, float]:
    set_global_seed(config.seed)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    save_run_snapshots(config, config.run_dir)
    device = resolve_device("auto")

    env = gym.make(config.env_id)
    obs, _ = env.reset(seed=config.seed)
    env.action_space.seed(config.seed)
    if not isinstance(env.action_space, Discrete):
        raise TypeError("VPG implementation requires a discrete action space")

    obs_dim = int(np.asarray(obs).shape[0])
    n_actions = int(env.action_space.n)
    agent = VPGAgent(obs_dim, n_actions, config, device)
    last_eval_return = 0.0
    last_policy_loss = 0.0
    last_value_loss = 0.0
    episode_idx = 0

    with CsvLogger(
        config.run_dir / "metrics.csv",
        [
            "epoch",
            "episode",
            "episode_return",
            "episode_length",
            "policy_loss",
            "value_loss",
            "eval_return",
        ],
    ) as logger:
        for epoch in range(config.epochs):
            all_obs: list[torch.Tensor] = []
            all_actions: list[torch.Tensor] = []
            all_returns: list[torch.Tensor] = []
            all_advantages: list[torch.Tensor] = []
            steps = 0

            while steps < config.steps_per_epoch:
                obs_buffer: list[NDArray[np.float32]] = []
                action_buffer: list[int] = []
                reward_buffer: list[float] = []
                value_buffer: list[float] = []
                log_prob_buffer: list[float] = []
                episode_return = 0.0
                episode_length = 0
                done = False

                while not done and steps < config.steps_per_epoch:
                    obs_array = np.asarray(obs, dtype=np.float32)
                    action, log_prob, value = agent.act(obs_array)
                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    obs_buffer.append(obs_array)
                    action_buffer.append(action)
                    reward_buffer.append(float(reward))
                    value_buffer.append(value)
                    log_prob_buffer.append(log_prob)

                    obs = next_obs
                    episode_return += float(reward)
                    episode_length += 1
                    steps += 1

                _append_episode_tensors(
                    all_obs,
                    all_actions,
                    all_returns,
                    all_advantages,
                    obs_buffer,
                    action_buffer,
                    reward_buffer,
                    value_buffer,
                    log_prob_buffer,
                    config.gamma,
                    device,
                )
                logger.log(
                    {
                        "epoch": epoch,
                        "episode": episode_idx,
                        "episode_return": episode_return,
                        "episode_length": episode_length,
                        "policy_loss": last_policy_loss,
                        "value_loss": last_value_loss,
                        "eval_return": last_eval_return,
                    }
                )
                episode_idx += 1

                if done:
                    obs, _ = env.reset(seed=config.seed + episode_idx)

            obs_batch = torch.cat(all_obs)
            actions_batch = torch.cat(all_actions)
            returns_batch = torch.cat(all_returns)
            advantages_batch = torch.cat(all_advantages)
            advantages_batch = (advantages_batch - advantages_batch.mean()) / (
                advantages_batch.std(unbiased=False) + 1e-8
            )

            update_metrics = agent.update(
                obs_batch,
                actions_batch,
                returns_batch,
                advantages_batch,
            )
            last_policy_loss = update_metrics["policy_loss"]
            last_value_loss = update_metrics["value_loss"]
            last_eval_return = evaluate(agent, config)["return_mean"]

    env.close()
    save_checkpoint(
        config.run_dir / "policy.pt",
        agent.policy,
        agent.policy_optimizer,
        metadata=CheckpointMetadata(step=config.epochs * config.steps_per_epoch, seed=config.seed),
        extra={"last_eval_return": last_eval_return},
    )
    save_checkpoint(
        config.run_dir / "value_function.pt",
        agent.value_function,
        agent.value_optimizer,
        metadata=CheckpointMetadata(step=config.epochs * config.steps_per_epoch, seed=config.seed),
        extra={"last_eval_return": last_eval_return},
    )

    example_input = torch.zeros(4, obs_dim, device=device)
    onnx_path = export_to_onnx(agent.policy, example_input, config.run_dir / "policy.onnx")
    result = compare_pytorch_onnx(agent.policy, onnx_path, example_input)
    if not result.passed:
        raise RuntimeError(f"ONNX consistency failed: {result}")

    return {
        "last_policy_loss": last_policy_loss,
        "last_value_loss": last_value_loss,
        "last_eval_return": last_eval_return,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VPG on CartPole.")
    parser.add_argument("--epochs", type=int, default=VPGConfig.epochs)
    parser.add_argument("--steps-per-epoch", type=int, default=VPGConfig.steps_per_epoch)
    parser.add_argument("--seed", type=int, default=VPGConfig.seed)
    parser.add_argument("--run-dir", type=Path, default=VPGConfig.run_dir)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = replace(
        VPGConfig(),
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        seed=args.seed,
        run_dir=args.run_dir,
    )
    metrics = train(config)
    print(metrics)


if __name__ == "__main__":
    main()
