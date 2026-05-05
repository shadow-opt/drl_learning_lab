from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Discrete

from drl_lab.algorithms.dqn.agent import DQNAgent, linear_epsilon
from drl_lab.algorithms.dqn.buffer import ReplayBuffer
from drl_lab.algorithms.dqn.config import DQNConfig
from drl_lab.algorithms.dqn.eval import evaluate
from drl_lab.common.checkpoint import CheckpointMetadata, save_checkpoint
from drl_lab.common.device import resolve_device
from drl_lab.common.experiment import save_run_snapshots
from drl_lab.common.export import export_to_onnx
from drl_lab.common.logging import CsvLogger
from drl_lab.common.onnx_check import compare_pytorch_onnx
from drl_lab.common.seed import set_global_seed


def train(config: DQNConfig) -> dict[str, float]:
    set_global_seed(config.seed)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    save_run_snapshots(config, config.run_dir)
    device = resolve_device("auto")

    env = gym.make(config.env_id)
    obs, _ = env.reset(seed=config.seed)
    env.action_space.seed(config.seed)

    obs_dim = int(np.asarray(obs).shape[0])
    if not isinstance(env.action_space, Discrete):
        raise TypeError("DQN requires a discrete action space")
    n_actions = int(env.action_space.n)
    agent = DQNAgent(obs_dim, n_actions, config, device)
    buffer = ReplayBuffer(config.buffer_capacity, obs_dim, seed=config.seed)

    episode_return = 0.0
    episode_length = 0
    episode_idx = 0
    last_loss = 0.0
    last_eval_return = 0.0

    with CsvLogger(
        config.run_dir / "metrics.csv",
        ["step", "episode", "episode_return", "episode_length", "epsilon", "loss", "eval_return"],
    ) as logger:
        for step in range(config.total_steps):
            epsilon = linear_epsilon(step, config)
            action = agent.act(np.asarray(obs, dtype=np.float32), epsilon=epsilon)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.add(
                np.asarray(obs, dtype=np.float32),
                action,
                float(reward),
                np.asarray(next_obs, dtype=np.float32),
                done,
            )

            obs = next_obs
            episode_return += float(reward)
            episode_length += 1

            if done:
                logger.log(
                    {
                        "step": step,
                        "episode": episode_idx,
                        "episode_return": episode_return,
                        "episode_length": episode_length,
                        "epsilon": epsilon,
                        "loss": last_loss,
                        "eval_return": last_eval_return,
                    }
                )
                obs, _ = env.reset(seed=config.seed + episode_idx + 1)
                episode_return = 0.0
                episode_length = 0
                episode_idx += 1

            if (
                step >= config.learning_starts
                and step % config.train_frequency == 0
                and len(buffer) >= config.batch_size
            ):
                batch = buffer.sample(config.batch_size, device=device)
                last_loss = agent.update(batch)

            if step > 0 and step % config.target_update_frequency == 0:
                agent.sync_target()

            if step > 0 and step % config.eval_frequency == 0:
                eval_metrics = evaluate(agent, config)
                last_eval_return = eval_metrics["return_mean"]

    env.close()
    final_eval = evaluate(agent, config)
    last_eval_return = final_eval["return_mean"]

    save_checkpoint(
        config.run_dir / "q_network.pt",
        agent.q_network,
        agent.optimizer,
        metadata=CheckpointMetadata(step=config.total_steps, seed=config.seed),
        extra={"last_eval_return": last_eval_return},
    )

    example_input = torch.zeros(4, obs_dim, device=device)
    onnx_path = export_to_onnx(agent.q_network, example_input, config.run_dir / "q_network.onnx")
    result = compare_pytorch_onnx(agent.q_network, onnx_path, example_input)
    if not result.passed:
        raise RuntimeError(f"ONNX consistency failed: {result}")

    return {"last_loss": last_loss, "last_eval_return": last_eval_return}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN on CartPole.")
    parser.add_argument("--total-steps", type=int, default=DQNConfig.total_steps)
    parser.add_argument("--learning-starts", type=int, default=DQNConfig.learning_starts)
    parser.add_argument("--seed", type=int, default=DQNConfig.seed)
    parser.add_argument("--run-dir", type=Path, default=DQNConfig.run_dir)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = replace(
        DQNConfig(),
        total_steps=args.total_steps,
        learning_starts=args.learning_starts,
        seed=args.seed,
        run_dir=args.run_dir,
    )
    metrics = train(config)
    print(metrics)


if __name__ == "__main__":
    main()
