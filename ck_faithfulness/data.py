"""GPQA Diamond sampling, plus the jsonl files every stage reads and appends to."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from ck_faithfulness import LETTERS

DOMAINS = ("physics", "chemistry", "biology")


# ---------------------------------------------------------------- GPQA sample


def stable_rng(*parts: Any, seed: int) -> random.Random:
    payload = f"{seed}|" + "|".join(map(str, parts))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def normalize_domain(raw: str | None) -> str:
    text = (raw or "").strip().lower()
    return next((d for d in DOMAINS if d in text), text or "unknown")


def _cell(row: dict[str, Any], *names: str) -> str:
    for name in names:
        if row.get(name) is not None:
            return str(row[name]).strip()
    raise KeyError(f"Missing columns {names} in {list(row.keys())[:12]}")


def row_to_item(row: dict[str, Any], seed: int) -> dict[str, Any]:
    """Shuffle options and fix one wrong option as the hint target, both seeded per question."""
    correct = _cell(row, "Correct Answer")
    record_id = _cell(row, "Record ID")
    domain_raw = _cell(row, "High-level domain")
    subdomain = row.get("Subdomain") or domain_raw

    choices = [correct, *(_cell(row, f"Incorrect Answer {i}") for i in (1, 2, 3))]
    stable_rng("options", record_id, seed=seed).shuffle(choices)
    options = dict(zip(LETTERS, choices))
    correct_letter = next(letter for letter, text in options.items() if text == correct)
    wrong_letters = [letter for letter in LETTERS if letter != correct_letter]
    return {
        "question_id": record_id,
        "domain": normalize_domain(domain_raw),
        "domain_raw": domain_raw,
        "subject": subdomain.strip() or normalize_domain(domain_raw),
        "question": _cell(row, "Question"),
        "options": options,
        "correct_letter": correct_letter,
        "target": stable_rng("target", record_id, seed=seed).choice(wrong_letters),
        "correct_answer_text": correct,
    }


def largest_remainder_counts(sizes: dict[str, int], n: int) -> dict[str, int]:
    total = sum(sizes.values())
    if total < n:
        raise ValueError(f"Need {n} items but only {total} available")
    raw = {k: n * v / total for k, v in sizes.items()}
    counts = {k: int(v) for k, v in raw.items()}
    order = sorted(raw, key=lambda k: (raw[k] - counts[k], sizes[k], k), reverse=True)
    for key in order[: n - sum(counts.values())]:
        counts[key] += 1
    return counts


def stratified_sample(items: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        by_domain.setdefault(item["domain"], []).append(item)
    counts = largest_remainder_counts({k: len(v) for k, v in by_domain.items()}, n)
    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    for domain, k in sorted(counts.items()):
        pool = list(by_domain[domain])
        rng.shuffle(pool)
        sampled.extend(pool[:k])
    rng.shuffle(sampled)
    return sampled


def prepare_sample(out_path: Path, n: int, seed: int) -> list[dict[str, Any]]:
    from datasets import load_dataset

    try:
        rows = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    except Exception as exc:  # noqa: BLE001 - gated dataset: explain how to get access
        raise RuntimeError(
            "Failed to load Idavidrein/gpqa (gpqa_diamond). Accept the terms at "
            "https://huggingface.co/datasets/Idavidrein/gpqa and set HF_TOKEN or run `huggingface-cli login`."
        ) from exc
    sample = stratified_sample([row_to_item(dict(r), seed=seed) for r in rows], n=n, seed=seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "n": len(sample),
        "source": "Idavidrein/gpqa",
        "config": "gpqa_diamond",
        "domain_counts": dict(sorted(Counter(item["domain"] for item in sample).items())),
        "items": sample,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return sample


def load_sample(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))["items"]


# ---------------------------------------------------------------- jsonl results

Key = tuple[str, str, str]  # (model, question_id, hint_type)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def latest_rows(path: Path) -> dict[Key, dict[str, Any]]:
    """Latest non-error row per (model, question_id, hint_type). Later lines win."""
    return {
        (row["model"], row["question_id"], row["hint_type"]): row
        for row in load_jsonl(path)
        if not row.get("error")
    }


def completed_keys(path: Path, *, redo_truncated_below: int | None = None) -> set[Key]:
    """Keys that need no further generation.

    With redo_truncated_below=N, a row cut off by num_predict counts as not done
    unless it was already generated with num_predict >= N.
    """
    done = set()
    for key, row in latest_rows(path).items():
        truncated = row.get("done_reason") == "length"
        budget = (row.get("gen_options") or {}).get("num_predict") or 0
        if redo_truncated_below is not None and truncated and budget < redo_truncated_below:
            continue
        done.add(key)
    return done


def load_generations(cfg, model_ids: list[str] | None = None) -> list[dict[str, Any]]:
    """All generation rows for the given model ids (default: every configured model)."""
    ids = model_ids or [m.id for m in cfg.models]
    return [row for mid in ids for row in load_jsonl(cfg.generation_path(mid))]
