"""Stage 1: generate baseline + hinted answers with a local Ollama model."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import requests

from ck_faithfulness import BASELINE, HINT_TYPES
from ck_faithfulness.config import ExperimentConfig, ModelSpec
from ck_faithfulness.data import append_jsonl, completed_keys, latest_rows, load_sample
from ck_faithfulness.parse import mentions_hint, parse_answer
from ck_faithfulness.prompts import ChatTurn, build_messages, generator_system_prompt


class OllamaClient:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.base = cfg.ollama_host

    def tags(self) -> list[str]:
        r = requests.get(f"{self.base}/api/tags", timeout=10)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    def has_model(self, model_id: str) -> bool:
        names = set(self.tags())
        return model_id in names or f"{model_id}:latest" in names

    def chat(self, model_id: str, turns: list[ChatTurn]) -> dict[str, Any]:
        messages = [{"role": "system", "content": generator_system_prompt(model_id)}]
        messages += [{"role": t.role, "content": t.content} for t in turns]
        r = requests.post(
            f"{self.base}/api/chat",
            json={
                "model": model_id,
                "messages": messages,
                "stream": False,
                "think": self.cfg.think,
                "keep_alive": self.cfg.keep_alive,
                "options": {
                    "temperature": self.cfg.temperature,
                    "seed": self.cfg.ollama_seed,
                    "num_ctx": self.cfg.num_ctx,
                    "num_predict": self.cfg.num_predict,
                },
            },
            timeout=self.cfg.timeout_s,
        )
        if r.status_code >= 400:
            raise RuntimeError(f"Ollama {r.status_code}: {r.text[:500]}")
        data = r.json()
        message = data.get("message") or {}
        return {
            "raw_response": message.get("content") or "",
            "thinking": message.get("thinking") or "",
            "eval_count": data.get("eval_count"),
            "eval_duration": data.get("eval_duration"),
            "prompt_eval_count": data.get("prompt_eval_count"),
            "total_duration": data.get("total_duration"),
            "done_reason": data.get("done_reason"),
        }

    def unload(self, model_id: str) -> None:
        """Free VRAM so the next generator or the judge can load."""
        try:
            requests.post(
                f"{self.base}/api/chat",
                json={"model": model_id, "messages": [{"role": "user", "content": "ping"}],
                      "keep_alive": 0, "options": {"num_predict": 1}},
                timeout=30,
            )
        except requests.RequestException:
            pass

    def require(self, model_id: str) -> None:
        if not self.has_model(model_id):
            raise SystemExit(
                f"Ollama does not have {model_id}. Download it first (not during a run):\n"
                f"  ollama pull {model_id}"
            )


def _messages(hint_type: str, item: dict[str, Any]) -> list[ChatTurn]:
    return build_messages(
        hint_type,
        question=item["question"],
        options=item["options"],
        target=item["target"],
        subject=item["subject"],
    )


def generate_model(
    cfg: ExperimentConfig,
    spec: ModelSpec,
    *,
    limit: int | None = None,
    dry_run: bool = False,
    redo_truncated: bool = False,
) -> dict[str, int]:
    items = load_sample(cfg.sample_path)[:limit]
    out_path = cfg.generation_path(spec.id)
    # A regenerated row is appended; readers take the latest row per key.
    done = completed_keys(out_path, redo_truncated_below=cfg.num_predict if redo_truncated else None)
    jobs = [(item, hint) for item in items for hint in (BASELINE, *HINT_TYPES)]
    remaining = [(item, hint) for item, hint in jobs if (spec.id, item["question_id"], hint) not in done]
    stats = {"planned": len(jobs), "skipped": len(jobs) - len(remaining), "written": 0, "errors": 0}
    if dry_run:
        print(f"[dry-run] {spec.id}: {stats['skipped']} done, {len(remaining)} remaining")
        return stats
    if not remaining:
        print(f"[generate] {spec.id}: all {stats['planned']} calls already checkpointed")
        return stats

    client = OllamaClient(cfg)
    client.require(spec.id)
    t0 = time.time()
    try:
        for i, (item, hint_type) in enumerate(remaining, start=1):
            started = time.time()
            try:
                gen, error = client.chat(spec.id, _messages(hint_type, item)), None
            except Exception as exc:  # noqa: BLE001 - persist the failure and continue
                gen, error = {"raw_response": ""}, str(exc)
            elapsed = time.time() - started
            raw = gen["raw_response"]
            parsed, parse_method = (None, "error") if error else parse_answer(raw, gen.get("done_reason"))
            append_jsonl(
                out_path,
                {
                    "model": spec.id,
                    "family": spec.family,
                    "question_id": item["question_id"],
                    "domain": item["domain"],
                    "subject": item["subject"],
                    "hint_type": hint_type,
                    "target": item["target"],
                    "correct_letter": item["correct_letter"],
                    "parsed_answer": parsed,
                    "parse_method": parse_method,
                    "raw_response": raw,
                    "thinking": gen.get("thinking") or "",
                    "keyword_mentions_hint": hint_type != BASELINE and mentions_hint(hint_type, raw),
                    "n_chars": len(raw),
                    "eval_count": gen.get("eval_count"),
                    "eval_duration_ns": gen.get("eval_duration"),
                    "prompt_eval_count": gen.get("prompt_eval_count"),
                    "total_duration_ns": gen.get("total_duration"),
                    "done_reason": gen.get("done_reason"),
                    "elapsed_s": round(elapsed, 3),
                    "error": error,
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "gen_options": {
                        "num_ctx": cfg.num_ctx,
                        "num_predict": cfg.num_predict,
                        "temperature": cfg.temperature,
                        "seed": cfg.ollama_seed,
                        "think": cfg.think,
                        "quant": spec.quant,
                    },
                },
            )
            stats["errors" if error else "written"] += 1
            eta = (time.time() - t0) / i * (len(remaining) - i)
            print(
                f"[generate] {spec.id} {i}/{len(remaining)} {item['question_id']} {hint_type} "
                f"ans={parsed or 'NA'} {elapsed:.1f}s eta={eta / 60:.1f}m",
                flush=True,
            )
    finally:
        client.unload(spec.id)
    return stats


def verify_determinism(cfg: ExperimentConfig, spec: ModelSpec, n: int = 7) -> dict[str, int]:
    """Re-run n rows that stopped naturally and compare text. Writes nothing.

    --redo-truncated assumes greedy decoding makes the larger num_predict/num_ctx a no-op
    for rows that already stopped. Check that before regenerating only the truncated rows.
    """
    items = {item["question_id"]: item for item in load_sample(cfg.sample_path)}
    stopped = sorted(
        (r for r in latest_rows(cfg.generation_path(spec.id)).values()
         if r.get("done_reason") == "stop" and r["question_id"] in items),
        key=lambda r: (r["question_id"], r["hint_type"]),
    )
    # One row per condition first, so the check covers every prompt shape.
    first_per_hint = list({r["hint_type"]: r for r in reversed(stopped)}.values())
    picked = (first_per_hint + [r for r in stopped if r not in first_per_hint])[:n]

    stats = {"checked": 0, "identical": 0, "same_answer": 0}
    if not picked:
        print(f"[verify] {spec.id}: no stopped rows to check")
        return stats
    client = OllamaClient(cfg)
    client.require(spec.id)
    try:
        for row in picked:
            gen = client.chat(spec.id, _messages(row["hint_type"], items[row["question_id"]]))
            same = gen["raw_response"] == row["raw_response"]
            old_ans = parse_answer(row["raw_response"], row.get("done_reason"))[0]
            new_ans = parse_answer(gen["raw_response"], gen.get("done_reason"))[0]
            stats["checked"] += 1
            stats["identical"] += int(same)
            stats["same_answer"] += int(old_ans == new_ans)
            print(
                f"[verify] {spec.id} {row['question_id']} {row['hint_type']}: "
                f"{'identical' if same else 'DIFFERENT'} answer {old_ans}->{new_ans}",
                flush=True,
            )
    finally:
        client.unload(spec.id)
    return stats
