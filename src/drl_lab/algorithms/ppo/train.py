from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Discrete
from numpy.typing import NDArray

from drl_lab.algorithms.ppo.agent import PPOAgent
from drl_lab.algorithms.ppo.buffer import make_ppo_batch
from drl_lab.algorithms.ppo.config import PPOConfig
from drl_lab.algorithms.ppo.eval import evaluate
from drl_lab.common.checkpoint import CheckpointMetadata, save_checkpoint
from drl_lab.common.device import resolve_device
from drl_lab.common.experiment import save_eval_report, save_export_report, save_run_snapshots
from drl_lab.common.export import export_to_onnx
from drl_lab.common.logging import CsvLogger
from drl_lab.common.onnx_check import compare_pytorch_onnx
from drl_lab.common.seed import set_global_seed


def _append_episode_tensors(
    all_obs: list[torch.Tensor],
    all_actions: list[torch.Tensor],
    all_returns: list[torch.Tensor],
    all_advantages: list[torch.Tensor],
    all_old_log_probs: list[torch.Tensor],
    obs_buffer: list[NDArray[np.float32]],
    action_buffer: list[int],
    reward_buffer: list[float],
    value_buffer: list[float],
    done_buffer: list[float],
    log_prob_buffer: list[float],
    gamma: float,
    lam: float,
    device: torch.device,
) -> None:
    obs_tensor = torch.as_tensor(np.asarray(obs_buffer), dtype=torch.float32, device=device)
    action_tensor = torch.as_tensor(action_buffer, dtype=torch.int64, device=device)
    reward_tensor = torch.as_tensor(reward_buffer, dtype=torch.float32, device=device)
    done_tensor = torch.as_tensor(done_buffer, dtype=torch.float32, device=device)
    value_tensor = torch.as_tensor(value_buffer + [0.0], dtype=torch.float32, device=device)
    log_prob_tensor = torch.as_tensor(log_prob_buffer, dtype=torch.float32, device=device)

    batch = make_ppo_batch(
        obs_tensor,
        action_tensor,
        reward_tensor,
        value_tensor,
        done_tensor,
        log_prob_tensor,
        gamma=gamma,
        lam=lam,
        normalize_advantage=False,
    )
    all_obs.append(batch.obs)
    all_actions.append(batch.actions)
    all_returns.append(batch.returns)
    all_advantages.append(batch.advantages)
    all_old_log_probs.append(batch.old_log_probs)


def train(config: PPOConfig) -> dict[str, float]:
    set_global_seed(config.seed)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    save_run_snapshots(config, config.run_dir)
    device = resolve_device("auto")

    env = gym.make(config.env_id)
    obs, _ = env.reset(seed=config.seed)
    env.action_space.seed(config.seed)
    if not isinstance(env.action_space, Discrete):
        raise TypeError("PPO implementation requires a discrete action space")

    obs_dim = int(np.asarray(obs).shape[0])
    n_actions = int(env.action_space.n)
    agent = PPOAgent(obs_dim, n_actions, config, device)

    last_metrics = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "approx_kl": 0.0,
        "entropy": 0.0,
        "policy_updates": 0.0,
        "eval_return": 0.0,
    }
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
            "approx_kl",
            "entropy",
            "policy_updates",
            "eval_return",
        ],
    ) as logger:
        for epoch in range(config.epochs):
            all_obs: list[torch.Tensor] = []
            all_actions: list[torch.Tensor] = []
            all_returns: list[torch.Tensor] = []
            all_advantages: list[torch.Tensor] = []
            all_old_log_probs: list[torch.Tensor] = []
            steps = 0

            while steps < config.steps_per_epoch:
                obs_buffer: list[NDArray[np.float32]] = []
                action_buffer: list[int] = []
                reward_buffer: list[float] = []
                value_buffer: list[float] = []
                done_buffer: list[float] = []
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
                    done_buffer.append(float(done))
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
                    all_old_log_probs,
                    obs_buffer,
                    action_buffer,
                    reward_buffer,
                    value_buffer,
                    done_buffer,
                    log_prob_buffer,
                    config.gamma,
                    config.lam,
                    device,
                )
                logger.log(
                    {
                        "epoch": epoch,
                        "episode": episode_idx,
                        "episode_return": episode_return,
                        "episode_length": episode_length,
                        **last_metrics,
                    }
                )
                episode_idx += 1

                if done:
                    obs, _ = env.reset(seed=config.seed + episode_idx)

            obs_batch = torch.cat(all_obs)
            actions_batch = torch.cat(all_actions)
            returns_batch = torch.cat(all_returns)
            advantages_batch = torch.cat(all_advantages)
            old_log_probs_batch = torch.cat(all_old_log_probs)
            advantages_batch = (advantages_batch - advantages_batch.mean()) / (
                advantages_batch.std(unbiased=False) + 1e-8
            )

            update_metrics = agent.update(
                obs_batch,
                actions_batch,
                returns_batch,
                advantages_batch,
                old_log_probs_batch,
            )
            eval_return = evaluate(agent, config)["return_mean"]
            last_metrics = {**update_metrics, "eval_return": eval_return}

    env.close()
    step_count = config.epochs * config.steps_per_epoch
    save_eval_report(config.run_dir, {"return_mean": last_metrics["eval_return"]})
    save_checkpoint(
        config.run_dir / "policy.pt",
        agent.policy,
        agent.policy_optimizer,
        metadata=CheckpointMetadata(step=step_count, seed=config.seed),
        extra={"last_eval_return": last_metrics["eval_return"]},
    )
    save_checkpoint(
        config.run_dir / "value_function.pt",
        agent.value_function,
        agent.value_optimizer,
        metadata=CheckpointMetadata(step=step_count, seed=config.seed),
        extra={"last_eval_return": last_metrics["eval_return"]},
    )

    example_input = torch.zeros(4, obs_dim, device=device)
    onnx_path = export_to_onnx(agent.policy, example_input, config.run_dir / "policy.onnx")
    result = compare_pytorch_onnx(agent.policy, onnx_path, example_input)
    if not result.passed:
        raise RuntimeError(f"ONNX consistency failed: {result}")
    value_onnx_path = export_to_onnx(
        agent.value_function,
        example_input,
        config.run_dir / "value_function.onnx",
    )
    value_result = compare_pytorch_onnx(agent.value_function, value_onnx_path, example_input)
    if not value_result.passed:
        raise RuntimeError(f"value ONNX consistency failed: {value_result}")
    policy_output_shape = list(agent.policy(example_input).shape)
    value_output_shape = list(agent.value_function(example_input).shape)
    save_export_report(
        config.run_dir,
        {
            "policy": {
                "path": onnx_path,
                "input_names": ["input"],
                "output_names": ["output"],
                "input_shape": list(example_input.shape),
                "output_shape": policy_output_shape,
                "max_abs_diff": result.max_abs_diff,
                "mean_abs_diff": result.mean_abs_diff,
                "passed": result.passed,
            },
            "value_function": {
                "path": value_onnx_path,
                "input_names": ["input"],
                "output_names": ["output"],
                "input_shape": list(example_input.shape),
                "output_shape": value_output_shape,
                "max_abs_diff": value_result.max_abs_diff,
                "mean_abs_diff": value_result.mean_abs_diff,
                "passed": value_result.passed,
            },
        },
    )

    return {
        "last_policy_loss": last_metrics["policy_loss"],
        "last_value_loss": last_metrics["value_loss"],
        "last_approx_kl": last_metrics["approx_kl"],
        "last_entropy": last_metrics["entropy"],
        "last_eval_return": last_metrics["eval_return"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on CartPole.")
    parser.add_argument("--epochs", type=int, default=PPOConfig.epochs)
    parser.add_argument("--steps-per-epoch", type=int, default=PPOConfig.steps_per_epoch)
    parser.add_argument("--seed", type=int, default=PPOConfig.seed)
    parser.add_argument("--run-dir", type=Path, default=PPOConfig.run_dir)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = replace(
        PPOConfig(),
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        seed=args.seed,
        run_dir=args.run_dir,
    )
    metrics = train(config)
    print(metrics)


if __name__ == "__main__":
    main()
