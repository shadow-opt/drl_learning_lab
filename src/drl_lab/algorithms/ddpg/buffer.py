from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray


@dataclass(frozen=True)
class ContinuousReplayBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor


class ContinuousReplayBuffer:
    """Replay buffer for continuous-action algorithms.

    Stored shapes:
        obs: ``[capacity, obs_dim]``
        actions: ``[capacity, action_dim]``
        rewards: ``[capacity]``
        next_obs: ``[capacity, obs_dim]``
        dones: ``[capacity]``
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, seed: int = 0) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if obs_dim <= 0 or action_dim <= 0:
            raise ValueError("obs_dim and action_dim must be positive")

        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self._rng = np.random.default_rng(seed)
        self._obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)
        self._pos = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        obs: NDArray[np.float32],
        action: NDArray[np.float32],
        reward: float,
        next_obs: NDArray[np.float32],
        done: bool,
    ) -> None:
        if obs.shape != (self.obs_dim,) or next_obs.shape != (self.obs_dim,):
            raise ValueError("obs and next_obs must have shape [obs_dim]")
        if action.shape != (self.action_dim,):
            raise ValueError("action must have shape [action_dim]")

        self._obs[self._pos] = obs
        self._actions[self._pos] = action
        self._rewards[self._pos] = reward
        self._next_obs[self._pos] = next_obs
        self._dones[self._pos] = float(done)
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device | str = "cpu") -> ContinuousReplayBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self._size < batch_size:
            raise ValueError("not enough samples in replay buffer")

        indices = self._rng.choice(self._size, size=batch_size, replace=False)
        torch_device = torch.device(device)
        return ContinuousReplayBatch(
            obs=torch.as_tensor(self._obs[indices], device=torch_device),
            actions=torch.as_tensor(self._actions[indices], device=torch_device),
            rewards=torch.as_tensor(self._rewards[indices], device=torch_device),
            next_obs=torch.as_tensor(self._next_obs[indices], device=torch_device),
            dones=torch.as_tensor(self._dones[indices], device=torch_device),
        )
