from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GridWorldConfig:
    rows: int = 4
    cols: int = 4
    start_state: int = 0
    terminal_state: int = 15
    step_reward: float = -1.0
    terminal_reward: float = 0.0


class GridWorld:
    """Small deterministic GridWorld for tabular RL exercises."""

    actions = ("up", "right", "down", "left")

    def __init__(self, config: GridWorldConfig | None = None) -> None:
        self.config = config or GridWorldConfig()
        self.n_states = self.config.rows * self.config.cols
        self.n_actions = len(self.actions)

    def to_row_col(self, state: int) -> tuple[int, int]:
        self._validate_state(state)
        return divmod(state, self.config.cols)

    def to_state(self, row: int, col: int) -> int:
        return row * self.config.cols + col

    def is_terminal(self, state: int) -> bool:
        self._validate_state(state)
        return state == self.config.terminal_state

    def transition(self, state: int, action: int) -> tuple[int, float, bool]:
        self._validate_state(state)
        if action < 0 or action >= self.n_actions:
            raise ValueError(f"invalid action {action}")

        if self.is_terminal(state):
            return state, self.config.terminal_reward, True

        row, col = self.to_row_col(state)
        if action == 0:
            row = max(0, row - 1)
        elif action == 1:
            col = min(self.config.cols - 1, col + 1)
        elif action == 2:
            row = min(self.config.rows - 1, row + 1)
        elif action == 3:
            col = max(0, col - 1)

        next_state = self.to_state(row, col)
        done = self.is_terminal(next_state)
        reward = self.config.terminal_reward if done else self.config.step_reward
        return next_state, reward, done

    def _validate_state(self, state: int) -> None:
        if state < 0 or state >= self.n_states:
            raise ValueError(f"invalid state {state}")
