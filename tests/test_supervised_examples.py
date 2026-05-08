from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_module(relative_path: str, module_name: str) -> ModuleType:
    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_linear_regression_learns_small_problem(tmp_path) -> None:  # type: ignore[no-untyped-def]
    module = load_module(
        "curriculum/chapters/00_ml_foundations/code/linear_regression.py",
        "linear_regression",
    )
    config = module.LinearRegressionConfig(epochs=80, run_dir=tmp_path / "linear")
    _, final_loss = module.train(config)
    assert final_loss < 0.05
    assert (config.run_dir / "config.json").exists()
    assert (config.run_dir / "environment.json").exists()


def test_binary_classifier_learns_small_problem(tmp_path) -> None:  # type: ignore[no-untyped-def]
    module = load_module(
        "curriculum/chapters/00_ml_foundations/code/binary_classifier.py",
        "binary_classifier",
    )
    config = module.BinaryClassifierConfig(epochs=80, run_dir=tmp_path / "binary")
    _, final_accuracy = module.train(config)
    assert final_accuracy > 0.98
    assert (config.run_dir / "config.json").exists()
    assert (config.run_dir / "environment.json").exists()


def test_image_classifier_learns_small_problem(tmp_path) -> None:  # type: ignore[no-untyped-def]
    module = load_module(
        "curriculum/chapters/00_ml_foundations/code/image_classifier.py",
        "image_classifier",
    )
    config = module.ImageClassifierConfig(
        samples_per_class=12,
        epochs=5,
        run_dir=tmp_path / "image",
    )
    _, final_accuracy = module.train(config)
    assert final_accuracy > 0.7
    assert (config.run_dir / "config.json").exists()
    assert (config.run_dir / "environment.json").exists()
