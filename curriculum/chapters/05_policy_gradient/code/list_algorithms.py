from __future__ import annotations

from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    for path in sorted(root.glob("*/lab.md")):
        print(path.parent.name)


if __name__ == "__main__":
    main()
