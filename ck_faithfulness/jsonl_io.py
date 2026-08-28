from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def record_key(model: str, question_id: str, hint_type: str) -> tuple[str, str, str]:
    return (model, question_id, hint_type)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def completed_keys(path: Path) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    for row in load_jsonl(path):
        if row.get("error"):
            continue
        keys.add(record_key(row["model"], row["question_id"], row["hint_type"]))
    return keys


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def iter_jsonl(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        yield from load_jsonl(path)
