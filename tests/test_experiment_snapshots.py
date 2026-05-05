from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from drl_lab.common.experiment import save_run_snapshots


@dataclass(frozen=True)
class TinyConfig:
    seed: int
    run_dir: Path


def test_save_run_snapshots_writes_config_and_environment(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = TinyConfig(seed=7, run_dir=tmp_path)

    config_path, environment_path = save_run_snapshots(config, tmp_path)

    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    environment_payload = json.loads(environment_path.read_text(encoding="utf-8"))
    assert config_payload["seed"] == 7
    assert config_payload["run_dir"] == str(tmp_path)
    assert "python" in environment_payload
    assert "torch" in environment_payload
