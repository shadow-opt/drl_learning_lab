from __future__ import annotations

from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    mappings = sorted(root.glob("*/spinningup_mapping.md"))
    for mapping in mappings:
        print(mapping.relative_to(root))


if __name__ == "__main__":
    main()
