from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from drl_lab.common.experiment import save_eval_report, save_export_report, save_run_snapshots


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
    assert "onnxruntime" in environment_payload
    assert "gymnasium" in environment_payload


def test_save_eval_and_export_reports_write_stable_json(tmp_path) -> None:  # type: ignore[no-untyped-def]
    eval_path = save_eval_report(tmp_path, {"return_mean": 12.0})
    export_path = save_export_report(
        tmp_path,
        {
            "policy": {
                "path": tmp_path / "policy.onnx",
                "input_shape": [4, 3],
                "output_shape": [4, 2],
                "max_abs_diff": 0.0,
                "passed": True,
            }
        },
    )

    eval_payload = json.loads(eval_path.read_text(encoding="utf-8"))
    export_payload = json.loads(export_path.read_text(encoding="utf-8"))
    assert eval_payload["return_mean"] == 12.0
    assert export_payload["artifacts"]["policy"]["path"].endswith("policy.onnx")
    assert export_payload["artifacts"]["policy"]["passed"] is True
