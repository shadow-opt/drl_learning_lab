from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box

from drl_lab.algorithms.ddpg.agent import DDPGAgent
from drl_lab.algorithms.ddpg.buffer import ContinuousReplayBuffer
from drl_lab.algorithms.ddpg.config import DDPGConfig
from drl_lab.algorithms.ddpg.eval import evaluate
from drl_lab.common.checkpoint import CheckpointMetadata, save_checkpoint
from drl_lab.common.device import resolve_device
from drl_lab.common.experiment import save_eval_report, save_export_report, save_run_snapshots
from drl_lab.common.export import export_to_onnx, export_to_onnx_multi_input
from drl_lab.common.logging import CsvLogger
from drl_lab.common.onnx_check import compare_pytorch_onnx, compare_pytorch_onnx_multi_input
from drl_lab.common.seed import set_global_seed


def train(config: DDPGConfig) -> dict[str, float]:
    set_global_seed(config.seed)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    save_run_snapshots(config, config.run_dir)
    device = resolve_device("auto")
    rng = np.random.default_rng(config.seed)

    env = gym.make(config.env_id)
    obs, _ = env.reset(seed=config.seed)
    env.action_space.seed(config.seed)
    if not isinstance(env.action_space, Box):
        raise TypeError("DDPG requires a continuous Box action space")

    obs_dim = int(np.asarray(obs).shape[0])
    action_dim = int(env.action_space.shape[0])
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    if not np.allclose(action_high, -action_low):
        raise ValueError("DDPG actor assumes symmetric action bounds")
    action_limit = float(action_high.max())

    agent = DDPGAgent(obs_dim, action_dim, action_limit, config, device)
    buffer = ContinuousReplayBuffer(
        config.buffer_capacity,
        obs_dim,
        action_dim,
        seed=config.seed,
    )

    episode_return = 0.0
    episode_length = 0
    episode_idx = 0
    last_actor_loss = 0.0
    last_critic_loss = 0.0
    last_eval_return = 0.0

    with CsvLogger(
        config.run_dir / "metrics.csv",
        [
            "step",
            "episode",
            "episode_return",
            "episode_length",
            "actor_loss",
            "critic_loss",
            "eval_return",
        ],
    ) as logger:
        for step in range(config.total_steps):
            obs_array = np.asarray(obs, dtype=np.float32)
            if step < config.learning_starts:
                action = np.asarray(env.action_space.sample(), dtype=np.float32)
            else:
                action = agent.act(obs_array, noise_scale=config.exploration_noise, rng=rng)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_obs_array = np.asarray(next_obs, dtype=np.float32)
            buffer.add(obs_array, action, float(reward), next_obs_array, done)

            obs = next_obs
            episode_return += float(reward)
            episode_length += 1

            if (
                step >= config.learning_starts
                and step % config.train_frequency == 0
                and len(buffer) >= config.batch_size
            ):
                batch = buffer.sample(config.batch_size, device=device)
                update_metrics = agent.update(batch)
                last_actor_loss = update_metrics["actor_loss"]
                last_critic_loss = update_metrics["critic_loss"]

            if done:
                logger.log(
                    {
                        "step": step,
                        "episode": episode_idx,
                        "episode_return": episode_return,
                        "episode_length": episode_length,
                        "actor_loss": last_actor_loss,
                        "critic_loss": last_critic_loss,
                        "eval_return": last_eval_return,
                    }
                )
                episode_idx += 1
                obs, _ = env.reset(seed=config.seed + episode_idx)
                episode_return = 0.0
                episode_length = 0

            if step > 0 and step % config.eval_frequency == 0:
                eval_metrics = evaluate(agent, config)
                last_eval_return = eval_metrics["return_mean"]

    env.close()
    final_eval = evaluate(agent, config)
    last_eval_return = final_eval["return_mean"]
    save_eval_report(config.run_dir, final_eval)

    save_checkpoint(
        config.run_dir / "actor.pt",
        agent.actor,
        agent.actor_optimizer,
        metadata=CheckpointMetadata(step=config.total_steps, seed=config.seed),
        extra={"last_eval_return": last_eval_return},
    )
    save_checkpoint(
        config.run_dir / "critic.pt",
        agent.critic,
        agent.critic_optimizer,
        metadata=CheckpointMetadata(step=config.total_steps, seed=config.seed),
        extra={"last_eval_return": last_eval_return},
    )

    example_input = torch.zeros(4, obs_dim, device=device)
    onnx_path = export_to_onnx(agent.actor, example_input, config.run_dir / "actor.onnx")
    result = compare_pytorch_onnx(agent.actor, onnx_path, example_input)
    if not result.passed:
        raise RuntimeError(f"ONNX consistency failed: {result}")
    example_action = torch.zeros(4, action_dim, device=device)
    critic_onnx_path = export_to_onnx_multi_input(
        agent.critic,
        (example_input, example_action),
        config.run_dir / "critic.onnx",
        input_names=["obs", "actions"],
        output_names=["q"],
    )
    critic_result = compare_pytorch_onnx_multi_input(
        agent.critic,
        critic_onnx_path,
        (example_input, example_action),
    )
    if not critic_result.passed:
        raise RuntimeError(f"critic ONNX consistency failed: {critic_result}")
    actor_output_shape = list(agent.actor(example_input).shape)
    critic_output_shape = list(agent.critic(example_input, example_action).shape)
    save_export_report(
        config.run_dir,
        {
            "actor": {
                "path": onnx_path,
                "input_names": ["input"],
                "output_names": ["output"],
                "input_shape": list(example_input.shape),
                "output_shape": actor_output_shape,
                "max_abs_diff": result.max_abs_diff,
                "mean_abs_diff": result.mean_abs_diff,
                "passed": result.passed,
            },
            "critic": {
                "path": critic_onnx_path,
                "input_names": ["obs", "actions"],
                "output_names": ["q"],
                "input_shape": [list(example_input.shape), list(example_action.shape)],
                "output_shape": critic_output_shape,
                "max_abs_diff": critic_result.max_abs_diff,
                "mean_abs_diff": critic_result.mean_abs_diff,
                "passed": critic_result.passed,
            },
        },
    )

    return {
        "last_actor_loss": last_actor_loss,
        "last_critic_loss": last_critic_loss,
        "last_eval_return": last_eval_return,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DDPG on Pendulum.")
    parser.add_argument("--total-steps", type=int, default=DDPGConfig.total_steps)
    parser.add_argument("--learning-starts", type=int, default=DDPGConfig.learning_starts)
    parser.add_argument("--eval-episodes", type=int, default=DDPGConfig.eval_episodes)
    parser.add_argument("--seed", type=int, default=DDPGConfig.seed)
    parser.add_argument("--run-dir", type=Path, default=DDPGConfig.run_dir)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = replace(
        DDPGConfig(),
        total_steps=args.total_steps,
        learning_starts=args.learning_starts,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        run_dir=args.run_dir,
    )
    metrics = train(config)
    print(metrics)


if __name__ == "__main__":
    main()
