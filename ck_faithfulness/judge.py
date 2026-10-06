"""Stage 2: label influenced cases with gpt-oss-20b on a local llama-server (official labels)."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import requests

from ck_faithfulness.config import ExperimentConfig
from ck_faithfulness.data import append_jsonl, load_generations, load_jsonl
from ck_faithfulness.parse import parse_verdict
from ck_faithfulness.prompts import case_input_sha, judge_messages
from ck_faithfulness.report import influenced_cases


def judge_reachable(base_url: str, timeout_s: float = 5) -> bool:
    root = base_url.removesuffix("/v1")
    for url in (f"{root}/health", f"{root}/v1/models"):
        try:
            if requests.get(url, timeout=timeout_s).status_code < 500:
                return True
        except requests.RequestException:
            continue
    return False


def message_text(message: dict[str, Any] | None) -> str:
    """Final content first, then gpt-oss reasoning, so the content's verdict wins."""
    message = message or {}
    parts = [message.get(k) for k in ("content", "reasoning_content", "reasoning")]
    return "\n".join(p for p in parts if isinstance(p, str) and p.strip())


def call_judge(cfg: ExperimentConfig, messages: list[dict[str, str]]) -> dict[str, Any]:
    url = f"{cfg.judge_base_url}/chat/completions"
    payload: dict[str, Any] = {
        "model": cfg.judge_model,
        "temperature": cfg.judge_temperature,
        "max_tokens": cfg.judge_max_tokens,
        "messages": messages,
        "chat_template_kwargs": {"reasoning_effort": cfg.judge_reasoning_effort},
    }
    r = requests.post(url, json=payload, timeout=cfg.judge_timeout_s)
    if r.status_code == 400 and "chat_template_kwargs" in (r.text or "").lower():
        payload.pop("chat_template_kwargs")  # older llama-server builds reject the field
        r = requests.post(url, json=payload, timeout=cfg.judge_timeout_s)
    if r.status_code >= 400:
        raise RuntimeError(f"llama.cpp API {r.status_code}: {r.text[:800]}")
    data = r.json()
    text = message_text(((data.get("choices") or [{}])[0]).get("message"))
    verdict = parse_verdict(text)
    if verdict is None:
        raise RuntimeError(f"Unparseable judge output: {text[:800]!r}")
    return {"verdict": verdict, "raw_judge": text, "usage": data.get("usage") or {}}


def pending_judgments(cfg: ExperimentConfig, model_ids: list[str] | None) -> list[dict[str, Any]]:
    """Influenced cases with no verdict for their current judge input."""
    judged: set[tuple] = set()
    for model in model_ids or [m.id for m in cfg.models]:
        for row in load_jsonl(cfg.judgment_path(model)):
            if row.get("verdict") and not row.get("error"):
                judged.add((row["model"], row["question_id"], row["hint_type"], row.get("judge_input_sha")))
    pending = []
    for case in influenced_cases(load_generations(cfg, model_ids)):
        sha = case_input_sha(cfg, case)
        if (case["model"], case["question_id"], case["hint_type"], sha) not in judged:
            pending.append({**case, "judge_input_sha": sha})
    return pending


def run_judge(
    cfg: ExperimentConfig,
    *,
    model_ids: list[str] | None = None,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    pending = pending_judgments(cfg, model_ids)[:limit]
    stats = {"pending": len(pending), "written": 0, "errors": 0}
    if not pending:
        print("[judge] nothing to judge (no influenced cases, or all checkpointed)")
        return stats
    if dry_run or not judge_reachable(cfg.judge_base_url):
        reason = "dry-run" if dry_run else f"llama.cpp server down at {cfg.judge_base_url}"
        print(f"[judge] {reason}: {len(pending)} influenced cases waiting for {cfg.judge_model}.")
        for case in pending[:10]:
            print(
                f"  - {case['model']} {case['question_id']} {case['hint_type']} "
                f"{case['baseline_answer']}->{case['hinted_answer']}"
            )
        if len(pending) > 10:
            print(f"  ... {len(pending) - 10} more")
        return stats

    t0 = time.time()
    for i, case in enumerate(pending, start=1):
        try:
            result, error = call_judge(cfg, judge_messages(cfg, case)), None
        except Exception as exc:  # noqa: BLE001 - persist the failure and continue
            result, error = {"verdict": None, "raw_judge": "", "usage": {}}, str(exc)
        append_jsonl(
            cfg.judgment_path(case["model"]),
            {
                "model": case["model"],
                "family": case["family"],
                "question_id": case["question_id"],
                "hint_type": case["hint_type"],
                "target": case["target"],
                "baseline_answer": case["baseline_answer"],
                "hinted_answer": case["hinted_answer"],
                "verdict": result["verdict"],
                "raw_judge": result["raw_judge"],
                "judge_input_sha": case["judge_input_sha"],
                "judge_model": cfg.judge_model,
                "judge_temperature": cfg.judge_temperature,
                "reasoning_effort": cfg.judge_reasoning_effort,
                "usage": result["usage"],
                "error": error,
                "ts": datetime.now(timezone.utc).isoformat(),
            },
        )
        stats["errors" if error else "written"] += 1
        eta = (time.time() - t0) / i * (len(pending) - i)
        print(
            f"[judge] {i}/{len(pending)} {case['model']} {case['question_id']} "
            f"{case['hint_type']} {result['verdict'] or 'ERR'} eta={eta:.0f}s",
            flush=True,
        )
    return stats
