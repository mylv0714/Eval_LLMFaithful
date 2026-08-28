from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

from ck_faithfulness.config import ExperimentConfig
from ck_faithfulness.jsonl_io import append_jsonl, completed_keys, load_jsonl
from ck_faithfulness.llamacpp_judge import LlamaCppJudge, llamacpp_reachable
from ck_faithfulness.metrics import influenced_cases
from ck_faithfulness.parse import truncate_cot


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def generation_paths(cfg: ExperimentConfig, model_ids: list[str] | None) -> list[Path]:
    if model_ids:
        return [cfg.generation_path(mid) for mid in model_ids]
    return [cfg.generation_path(spec.id) for spec in cfg.models]


def collect_generation_rows(cfg: ExperimentConfig, model_ids: list[str] | None) -> list[dict]:
    rows = []
    for path in generation_paths(cfg, model_ids):
        rows.extend(load_jsonl(path))
    return rows


def pending_judgments(cfg: ExperimentConfig, model_ids: list[str] | None) -> list[dict]:
    rows = collect_generation_rows(cfg, model_ids)
    cases = influenced_cases(rows)
    pending = []
    for case in cases:
        path = cfg.judgment_path(case["model"])
        done = completed_keys(path)
        if (case["model"], case["question_id"], case["hint_type"]) in done:
            continue
        pending.append(case)
    return pending


def judge_available(cfg: ExperimentConfig) -> bool:
    return llamacpp_reachable(cfg.judge_base_url)


def build_judge(cfg: ExperimentConfig) -> LlamaCppJudge:
    return LlamaCppJudge(cfg)


def run_judge(
    cfg: ExperimentConfig,
    *,
    model_ids: list[str] | None = None,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    pending = pending_judgments(cfg, model_ids)
    if limit is not None:
        pending = pending[:limit]
    stats = {"pending": len(pending), "written": 0, "errors": 0}
    if not pending:
        print("[judge] nothing to judge (no influenced cases, or all checkpointed)")
        return stats
    if dry_run or not judge_available(cfg):
        reason = "dry-run" if dry_run else f"llama.cpp server down at {cfg.judge_base_url}"
        print(
            f"[judge] {reason}: {len(pending)} influenced cases waiting. "
            f"Official labels use {cfg.judge_family} via llama.cpp."
        )
        for case in pending[:10]:
            print(
                f"  - {case['model']} {case['question_id']} {case['hint_type']} "
                f"{case['baseline_answer']}->{case['hinted_answer']}"
            )
        if len(pending) > 10:
            print(f"  ... {len(pending) - 10} more")
        return stats

    judge = build_judge(cfg)
    t0 = time.time()
    with judge:
        for i, case in enumerate(pending, start=1):
            cot = truncate_cot(case["raw_response"], cfg.cot_head_chars, cfg.cot_tail_chars)
            path = cfg.judgment_path(case["model"])
            try:
                result = judge.classify(case["hint_type"], case["target"], cot)
                error = None
            except Exception as exc:  # noqa: BLE001
                result = {
                    "verdict": None,
                    "raw_judge": "",
                    "judge_model": cfg.judge_model,
                    "usage": {},
                    "backend": cfg.judge_backend,
                }
                error = str(exc)
            row = {
                "model": case["model"],
                "family": case["family"],
                "question_id": case["question_id"],
                "hint_type": case["hint_type"],
                "target": case["target"],
                "baseline_answer": case["baseline_answer"],
                "hinted_answer": case["hinted_answer"],
                "influenced": True,
                "verdict": result["verdict"],
                "faithful": (result["verdict"] == "YES") if result["verdict"] else None,
                "raw_judge": result["raw_judge"],
                "judge_family": cfg.judge_family,
                "judge_model": result["judge_model"],
                "judge_backend": result.get("backend") or cfg.judge_backend,
                "judge_temperature": cfg.judge_temperature,
                "reasoning_effort": cfg.judge_reasoning_effort,
                "usage": result["usage"],
                "keyword_mentions_hint": case.get("keyword_mentions_hint"),
                "n_chars": case.get("n_chars"),
                "error": error,
                "ts": _now(),
            }
            append_jsonl(path, row)
            if error:
                stats["errors"] += 1
            else:
                stats["written"] += 1
            eta = (time.time() - t0) / i * (len(pending) - i)
            print(
                f"[judge] {i}/{len(pending)} {case['model']} {case['question_id']} "
                f"{case['hint_type']} {result['verdict'] or 'ERR'} eta={eta:.0f}s",
                flush=True,
            )
    return stats


def judgment_dir_ready(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
