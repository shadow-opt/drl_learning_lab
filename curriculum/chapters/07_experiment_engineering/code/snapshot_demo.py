from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from drl_lab.common.experiment import save_run_snapshots


@dataclass(frozen=True)
class SnapshotDemoConfig:
    seed: int = 0
    lr: float = 0.001
    run_dir: Path = Path("experiments/runs/snapshot_demo")


def main() -> None:
    config = SnapshotDemoConfig()
    config_path, environment_path = save_run_snapshots(config, config.run_dir)
    print(f"config={config_path}")
    print(f"environment={environment_path}")


if __name__ == "__main__":
    main()
