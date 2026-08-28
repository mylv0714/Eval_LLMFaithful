from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from ck_faithfulness import BASELINE, HINT_TYPES
from ck_faithfulness.config import ExperimentConfig, ModelSpec
from ck_faithfulness.data import load_sample
from ck_faithfulness.hints import build_messages
from ck_faithfulness.jsonl_io import append_jsonl, completed_keys
from ck_faithfulness.mention import mentions_hint
from ck_faithfulness.ollama_client import OllamaClient
from ck_faithfulness.parse import extract_answer


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def planned_jobs(items: list[dict[str, Any]], model_id: str) -> list[dict[str, Any]]:
    jobs = []
    for item in items:
        jobs.append({"model": model_id, "hint_type": BASELINE, "item": item})
        for hint_type in HINT_TYPES:
            jobs.append({"model": model_id, "hint_type": hint_type, "item": item})
    return jobs


def generate_model(
    cfg: ExperimentConfig,
    spec: ModelSpec,
    *,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    items = load_sample(cfg.sample_path)
    if limit is not None:
        items = items[:limit]
    client = OllamaClient(cfg)
    out_path = cfg.generation_path(spec.id)
    done = completed_keys(out_path)
    jobs = planned_jobs(items, spec.id)
    remaining = [
        job
        for job in jobs
        if (job["model"], job["item"]["question_id"], job["hint_type"]) not in done
    ]
    stats = {"planned": len(jobs), "skipped": len(jobs) - len(remaining), "written": 0, "errors": 0}
    if dry_run:
        print(f"[dry-run] {spec.id}: {stats['skipped']} done, {len(remaining)} remaining")
        return stats
    if not remaining:
        print(f"[generate] {spec.id}: all {stats['planned']} calls already checkpointed")
        return stats
    if not client.has_model(spec.id):
        raise SystemExit(
            f"Ollama does not have {spec.id}. Pull it first, e.g.\n"
            f"  ollama pull {spec.id}\n"
            "Do not start a long run until the model is local."
        )

    t0 = time.time()
    try:
        for i, job in enumerate(remaining, start=1):
            item = job["item"]
            hint_type = job["hint_type"]
            turns = build_messages(
                None if hint_type == BASELINE else hint_type,
                question=item["question"],
                options=item["options"],
                target=item["target"],
                subject=item["subject"],
            )
            started = time.time()
            try:
                gen = client.chat(spec.id, turns)
                error = None
            except Exception as exc:  # noqa: BLE001 — persist and continue
                gen = {
                    "raw_response": "",
                    "thinking": "",
                    "eval_count": None,
                    "eval_duration": None,
                    "prompt_eval_count": None,
                    "total_duration": None,
                    "done_reason": None,
                }
                error = str(exc)
            elapsed = time.time() - started
            raw = gen["raw_response"]
            parsed = extract_answer(raw) if not error else None
            row = {
                "model": spec.id,
                "family": spec.family,
                "question_id": item["question_id"],
                "domain": item["domain"],
                "subject": item["subject"],
                "hint_type": hint_type,
                "target": item["target"],
                "correct_letter": item["correct_letter"],
                "parsed_answer": parsed,
                "raw_response": raw,
                "thinking": gen.get("thinking") or "",
                "keyword_mentions_hint": (
                    False
                    if hint_type == BASELINE
                    else mentions_hint(hint_type, raw)
                ),
                "n_chars": len(raw or ""),
                "eval_count": gen.get("eval_count"),
                "eval_duration_ns": gen.get("eval_duration"),
                "prompt_eval_count": gen.get("prompt_eval_count"),
                "total_duration_ns": gen.get("total_duration"),
                "done_reason": gen.get("done_reason"),
                "elapsed_s": round(elapsed, 3),
                "error": error,
                "ts": _now(),
                "gen_options": {
                    "num_ctx": cfg.num_ctx,
                    "num_predict": cfg.num_predict,
                    "temperature": cfg.temperature,
                    "seed": cfg.ollama_seed,
                    "think": cfg.think,
                    "quant": spec.quant,
                },
            }
            append_jsonl(out_path, row)
            if error:
                stats["errors"] += 1
            else:
                stats["written"] += 1
            eta = (time.time() - t0) / i * (len(remaining) - i)
            print(
                f"[generate] {spec.id} {i}/{len(remaining)} "
                f"{item['question_id']} {hint_type} "
                f"ans={parsed or 'NA'} {elapsed:.1f}s eta={eta/60:.1f}m",
                flush=True,
            )
    finally:
        client.unload(spec.id)
    return stats
