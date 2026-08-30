"""Atomic writers for experiment artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write a complete JSON document before replacing the destination."""

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def atomic_write_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, object]],
) -> None:
    """Write JSON objects one per line before replacing the destination."""

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    partial.replace(path)
