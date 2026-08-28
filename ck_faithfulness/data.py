from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

from ck_faithfulness import LETTERS, N_QUESTIONS, SEED

DOMAIN_ALIASES = {
    "physics": "physics",
    "chemistry": "chemistry",
    "biology": "biology",
    "organic chemistry": "chemistry",
    "inorganic chemistry": "chemistry",
}


def stable_rng(*parts: Any, seed: int = SEED) -> random.Random:
    payload = f"{seed}|" + "|".join(map(str, parts))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def normalize_domain(raw: str | None) -> str:
    text = (raw or "").strip().lower()
    if text in DOMAIN_ALIASES:
        return DOMAIN_ALIASES[text]
    for key, value in DOMAIN_ALIASES.items():
        if key in text:
            return value
    return text or "unknown"


def _cell(row: dict[str, Any], *names: str) -> str:
    for name in names:
        if name in row and row[name] is not None:
            return str(row[name]).strip()
    lower = {str(k).lower(): v for k, v in row.items()}
    for name in names:
        key = name.lower()
        if key in lower and lower[key] is not None:
            return str(lower[key]).strip()
    raise KeyError(f"Missing columns {names} in {list(row.keys())[:12]}")


def row_to_item(row: dict[str, Any], seed: int = SEED) -> dict[str, Any]:
    question = _cell(row, "Question", "Pre-Revision Question")
    correct = _cell(row, "Correct Answer")
    wrongs = [
        _cell(row, "Incorrect Answer 1"),
        _cell(row, "Incorrect Answer 2"),
        _cell(row, "Incorrect Answer 3"),
    ]
    record_id = _cell(row, "Record ID", "record_id", "id")
    domain_raw = _cell(row, "High-level domain", "High-level Domain", "domain")
    try:
        subdomain = _cell(row, "Subdomain", "Sub-domain")
    except KeyError:
        subdomain = domain_raw

    domain = normalize_domain(domain_raw)
    rng = stable_rng("options", record_id, seed=seed)
    choices = [correct, *wrongs]
    rng.shuffle(choices)
    options = {letter: text for letter, text in zip(LETTERS, choices)}
    correct_letter = next(letter for letter, text in options.items() if text == correct)
    wrong_letters = [letter for letter in LETTERS if letter != correct_letter]
    target = stable_rng("target", record_id, seed=seed).choice(wrong_letters)
    subject = subdomain.strip() if subdomain.strip() else domain
    return {
        "question_id": record_id,
        "domain": domain,
        "domain_raw": domain_raw,
        "subject": subject,
        "question": question,
        "options": options,
        "correct_letter": correct_letter,
        "target": target,
        "correct_answer_text": correct,
    }


def largest_remainder_counts(sizes: dict[str, int], n: int) -> dict[str, int]:
    total = sum(sizes.values())
    if total < n:
        raise ValueError(f"Need {n} items but only {total} available")
    raw = {k: (n * v / total) for k, v in sizes.items()}
    base = {k: int(v) for k, v in raw.items()}
    leftover = n - sum(base.values())
    order = sorted(raw, key=lambda k: (raw[k] - base[k], sizes[k], k), reverse=True)
    for key in order[:leftover]:
        base[key] += 1
    return base


def stratified_sample(
    items: list[dict[str, Any]],
    n: int = N_QUESTIONS,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        by_domain.setdefault(item["domain"], []).append(item)
    counts = largest_remainder_counts({k: len(v) for k, v in by_domain.items()}, n)
    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    for domain, k in sorted(counts.items()):
        pool = list(by_domain[domain])
        rng.shuffle(pool)
        if k > len(pool):
            raise ValueError(f"Domain {domain}: need {k}, have {len(pool)}")
        sampled.extend(pool[:k])
    rng.shuffle(sampled)
    return sampled


def load_gpqa_diamond_rows() -> list[dict[str, Any]]:
    from datasets import load_dataset

    last_error: Exception | None = None
    for kwargs in (
        {"path": "Idavidrein/gpqa", "name": "gpqa_diamond", "split": "train"},
        {"path": "Idavidrein/gpqa", "name": "gpqa_diamond"},
    ):
        try:
            ds = load_dataset(**kwargs)
            if hasattr(ds, "keys"):
                split = ds["train"] if "train" in ds else ds[next(iter(ds.keys()))]
            else:
                split = ds
            return [dict(row) for row in split]
        except Exception as exc:  # noqa: BLE001 — try next loader variant
            last_error = exc
    raise RuntimeError(
        "Failed to load Idavidrein/gpqa (gpqa_diamond). "
        "Accept the terms at https://huggingface.co/datasets/Idavidrein/gpqa "
        "and run `huggingface-cli login`."
    ) from last_error


def prepare_sample(
    out_path: Path,
    n: int = N_QUESTIONS,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    rows = load_gpqa_diamond_rows()
    items = [row_to_item(row, seed=seed) for row in rows]
    sample = stratified_sample(items, n=n, seed=seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "n": len(sample),
        "source": "Idavidrein/gpqa",
        "config": "gpqa_diamond",
        "domain_counts": _counts(sample),
        "items": sample,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return sample


def load_sample(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["items"]


def _counts(items: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        out[item["domain"]] = out.get(item["domain"], 0) + 1
    return dict(sorted(out.items()))


def looks_like_question_id(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_.-]+", value or ""))
