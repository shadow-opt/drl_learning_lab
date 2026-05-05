from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class CsvLogger:
    """Minimal CSV logger for small experiments."""

    def __init__(self, path: str | Path, fieldnames: list[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()

    def log(self, row: dict[str, Any]) -> None:
        clean_row = {key: row.get(key, "") for key in self.fieldnames}
        self._writer.writerow(clean_row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> CsvLogger:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
