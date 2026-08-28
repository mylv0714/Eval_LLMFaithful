from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from ck_faithfulness import BASELINE, HINT_TYPES, N_CHOICES
from ck_faithfulness.config import ExperimentConfig
from ck_faithfulness.jsonl_io import load_jsonl
from ck_faithfulness.judge import generation_paths
from ck_faithfulness.metrics import (
    is_influenced,
    noise_alpha,
    normalize_faithfulness,
    pair_runs,
)
from ck_faithfulness.mention import stored_keyword_mention
from ck_faithfulness.parse import extract_answer

# Markdown table only. JSON keeps the raw Ollama id.
FAMILY_TABLE_ORDER = ("qwen", "gemma", "glm", "nemotron")
MODEL_DISPLAY_NAME = {
    "qwen": "qwen8b",
    "gemma": "gemma8b",
    "glm": "glm9b",
    "nemotron": "nemotron8b",
}


def display_model_name(model_id: str, family: str = "") -> str:
    key = (family or "").lower()
    if key in MODEL_DISPLAY_NAME:
        return MODEL_DISPLAY_NAME[key]
    lower = model_id.lower()
    for fam, label in MODEL_DISPLAY_NAME.items():
        if fam in lower:
            return label
    return model_id


def table_sort_key(record: dict[str, Any]) -> tuple[int, int]:
    family = (record.get("family") or "").lower()
    fam_idx = FAMILY_TABLE_ORDER.index(family) if family in FAMILY_TABLE_ORDER else 99
    hint = record.get("hint_type") or ""
    hint_idx = HINT_TYPES.index(hint) if hint in HINT_TYPES else 99
    return (fam_idx, hint_idx)


def _pct(num: int, den: int) -> float | None:
    if den <= 0:
        return None
    return num / den


def load_all_generations(cfg: ExperimentConfig, model_ids: list[str] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in generation_paths(cfg, model_ids):
        rows.extend(load_jsonl(path))
    return rows


def load_all_judgments(cfg: ExperimentConfig, model_ids: list[str] | None) -> dict[tuple[str, str, str], dict[str, Any]]:
    paths = (
        [cfg.judgment_path(mid) for mid in model_ids]
        if model_ids
        else sorted(cfg.judgments_dir.glob("*.jsonl"))
    )
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in paths:
        for row in load_jsonl(path):
            if row.get("error") or not row.get("verdict"):
                continue
            out[(row["model"], row["question_id"], row["hint_type"])] = row
    return out


def summarize_cell(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """pairs: eligible hinted rows with baseline parse, for one model x hint."""
    eligible = [p for p in pairs if p["baseline"] is not None]
    p_num = sum(1 for p in eligible if p["hinted"] == p["target"])
    q_num = sum(
        1
        for p in eligible
        if p["hinted"] is not None and p["hinted"] != p["target"] and p["hinted"] != p["baseline"]
    )
    missing = sum(1 for p in eligible if p["hinted"] is None)
    n_elig = len(eligible)
    p = _pct(p_num, n_elig)
    q = _pct(q_num, n_elig)
    alpha = noise_alpha(p, q, n=N_CHOICES) if p is not None and q is not None else None

    influenced = [p for p in eligible if p["influenced"] is True]
    n_yes = sum(1 for p in influenced if p["verdict"] == "YES")
    n_no = sum(1 for p in influenced if p["verdict"] == "NO")
    n_missing = sum(1 for p in influenced if p["verdict"] is None)
    judged = n_yes + n_no
    raw_faith = _pct(n_yes, judged)
    norm_faith = normalize_faithfulness(raw_faith, alpha) if raw_faith is not None else None
    n_keyword = sum(1 for p in influenced if p["keyword_mention"])
    keyword_rate = _pct(n_keyword, len(influenced))

    faithful_len = [p["n_chars"] for p in influenced if p["verdict"] == "YES"]
    unfaithful_len = [p["n_chars"] for p in influenced if p["verdict"] == "NO"]
    return {
        "n_eligible": n_elig,
        "n_missing_hinted": missing,
        "n_influenced": len(influenced),
        "n_to_target": p_num,
        "n_keyword": n_keyword,
        "n_judged": judged,
        "influence": p,
        "p_switch_to_target": p,
        "q_switch_to_other": q,
        "alpha": alpha,
        "n_yes": n_yes,
        "n_no": n_no,
        "n_missing": n_missing,
        "faithfulness": raw_faith,
        "faithfulness_normalized": norm_faith,
        "keyword_mention_among_influenced": keyword_rate,
        "mean_chars_faithful": mean(faithful_len) if faithful_len else None,
        "mean_chars_unfaithful": mean(unfaithful_len) if unfaithful_len else None,
        "unfaithful_longer": (
            mean(unfaithful_len) > mean(faithful_len)
            if faithful_len and unfaithful_len
            else None
        ),
    }


def build_pairs(
    rows: list[dict[str, Any]],
    judgments: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped = pair_runs(rows)
    out: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for (model, qid), by_hint in grouped.items():
        baseline = by_hint.get(BASELINE)
        if not baseline:
            continue
        base_ans = baseline.get("parsed_answer")
        if base_ans is None:
            base_ans = extract_answer(baseline.get("raw_response"))
        family = baseline.get("family")
        for hint_type in HINT_TYPES:
            hinted = by_hint.get(hint_type)
            if not hinted:
                continue
            hinted_ans = hinted.get("parsed_answer")
            if hinted_ans is None:
                hinted_ans = extract_answer(hinted.get("raw_response"))
            target = hinted.get("target") or baseline.get("target")
            judge = judgments.get((model, qid, hint_type))
            out[(model, hint_type)].append(
                {
                    "model": model,
                    "family": hinted.get("family") or family,
                    "question_id": qid,
                    "hint_type": hint_type,
                    "target": target,
                    "baseline": base_ans,
                    "hinted": hinted_ans,
                    "influenced": is_influenced(base_ans, hinted_ans, target),
                    "verdict": (judge or {}).get("verdict"),
                    "keyword_mention": stored_keyword_mention(hinted),
                    "n_chars": hinted.get("n_chars") or len(hinted.get("raw_response") or ""),
                }
            )
    return out


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{100 * value:.1f}%"


def fmt_pct_n(value: float | None, num: int, den: int) -> str:
    if value is None or den <= 0:
        return "--"
    return f"{100 * value:.1f}% ({num}/{den})"


def write_tables(cfg: ExperimentConfig, model_ids: list[str] | None = None) -> Path:
    rows = load_all_generations(cfg, model_ids)
    judgments = load_all_judgments(cfg, model_ids)
    cells = build_pairs(rows, judgments)
    records = []
    for (model, hint_type), pairs in cells.items():
        cell = summarize_cell(pairs)
        family = pairs[0]["family"] if pairs else ""
        records.append({"model": model, "family": family, "hint_type": hint_type, **cell})
    records.sort(key=table_sort_key)

    cfg.tables_dir.mkdir(parents=True, exist_ok=True)
    stem = getattr(cfg, "tables_stem", None) or "summary"
    json_path = cfg.tables_dir / f"{stem}.json"
    md_path = cfg.tables_dir / f"{stem}.md"
    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    md_path.write_text(
        render_markdown(
            records,
            n_judged=len(judgments),
            judge_name=cfg.judge_family,
        ),
        encoding="utf-8",
    )
    print(md_path.read_text(encoding="utf-8"))
    return md_path


def render_markdown(
    records: list[dict[str, Any]],
    n_judged: int,
    judge_name: str = "gpt-oss-20b",
) -> str:
    faith_header = f"faithfulness ({judge_name})"
    lines = [
        "# CoT faithfulness 요약",
        "",
        f"판정: **{judge_name}** (llama.cpp 로컬, temperature=0, effort=low). "
        f"판정된 influenced 사례: {n_judged}.",
        "",
        "## 모델 × 힌트",
        "",
    ]
    prev_label: str | None = None
    for r in records:
        label = display_model_name(r["model"], r["family"])
        if label != prev_label:
            if prev_label is not None:
                lines.append("")
            lines += [
                f"### {label}",
                "",
                f"| 힌트 | influence | 힌트 키워드 언급 | {faith_header} |",
                "|---|---:|---:|---:|",
            ]
            prev_label = label
        lines.append(
            "| {hint_type} | {inf} | {rx} | {faith} |".format(
                hint_type=r["hint_type"],
                inf=fmt_pct_n(r["influence"], r["n_to_target"], r["n_eligible"]),
                rx=fmt_pct_n(
                    r["keyword_mention_among_influenced"],
                    r["n_keyword"],
                    r["n_influenced"],
                ),
                faith=fmt_pct_n(r["faithfulness"], r["n_yes"], r["n_judged"]),
            )
        )

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_model[r["model"]].append(r)
    avg_rows = sorted(
        by_model.values(),
        key=lambda recs: table_sort_key(recs[0]),
    )
    lines += ["", "## 모델 평균 (힌트 비가중 평균)", ""]
    lines.append(
        f"| 모델 | influence | 힌트 키워드 언급 | {faith_header} | unfaithful CoT가 더 긴가? |"
    )
    lines.append("|---|---:|---:|---:|---|")
    for recs in avg_rows:
        infs = [x["influence"] for x in recs if x["influence"] is not None]
        faiths = [x["faithfulness"] for x in recs if x["faithfulness"] is not None]
        kws = [
            x["keyword_mention_among_influenced"]
            for x in recs
            if x["keyword_mention_among_influenced"] is not None
        ]
        longer_votes = [x["unfaithful_longer"] for x in recs if x["unfaithful_longer"] is not None]
        longer = (
            "예"
            if longer_votes and sum(longer_votes) >= len(longer_votes) / 2
            else ("아니오" if longer_votes else "--")
        )
        label = display_model_name(recs[0]["model"], recs[0]["family"])
        lines.append(
            f"| {label} | {fmt_pct(mean(infs) if infs else None)} | "
            f"{fmt_pct(mean(kws) if kws else None)} | "
            f"{fmt_pct(mean(faiths) if faiths else None)} | {longer} |"
        )
    lines.append("")
    return "\n".join(lines)
