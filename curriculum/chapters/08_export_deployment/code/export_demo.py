from __future__ import annotations

import importlib.util
from pathlib import Path


def main() -> None:
    legacy_path = Path(__file__).resolve().parents[1] / "export_demo.py"
    spec = importlib.util.spec_from_file_location("export_demo", legacy_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {legacy_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()
