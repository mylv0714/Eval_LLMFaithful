"""Detailed analysis behind docs/mmlu_analysis.md and docs/gpqa_analysis.md.

Reads the generations and current (non-stale) judge verdicts and prints compact sections:
per-model metrics with question-level cluster-bootstrap 95% CIs, gemma4 vs others (Holm),
the difference from the other dataset, per-hint pooling, baseline-correct split,
subject category, CoT length, mention position, judge ROLE, judge vs keyword agreement.

    python scripts/deep_analysis.py --config configs/mmlu.yaml --compare configs/gpqa.yaml
    python scripts/deep_analysis.py --config configs/gpqa.yaml --compare configs/mmlu.yaml
"""

from __future__ import annotations

import argparse
import random
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ck_faithfulness import BASELINE, HINT_TYPES  # noqa: E402
from ck_faithfulness.config import load_config  # noqa: E402
from ck_faithfulness.data import load_generations, load_jsonl  # noqa: E402
from ck_faithfulness.parse import QUOTE_RE, row_answer  # noqa: E402
from ck_faithfulness.prompts import case_input_sha  # noqa: E402
from ck_faithfulness.report import hint_pairs, noise_alpha, pair_runs  # noqa: E402

B = 2000  # bootstrap resamples
REF_MODEL = "gemma4-8b"

# MMLU's four official categories (Hendrycks et al.). GPQA uses its own domain.
MMLU_CATS = {
    "STEM": "abstract_algebra astronomy college_biology college_chemistry college_computer_science college_mathematics "
            "college_physics computer_security conceptual_physics electrical_engineering elementary_mathematics "
            "high_school_biology high_school_chemistry high_school_computer_science high_school_mathematics "
            "high_school_physics high_school_statistics machine_learning",
    "Humanities": "formal_logic high_school_european_history high_school_us_history high_school_world_history "
                  "international_law jurisprudence logical_fallacies moral_disputes moral_scenarios philosophy "
                  "prehistory professional_law world_religions",
    "Social": "econometrics high_school_geography high_school_government_and_politics high_school_macroeconomics "
              "high_school_microeconomics high_school_psychology human_sexuality professional_psychology "
              "public_relations security_studies sociology us_foreign_policy",
    "Other": "anatomy business_ethics clinical_knowledge college_medicine global_facts human_aging management "
             "marketing medical_genetics miscellaneous nutrition professional_accounting professional_medicine virology",
}
MMLU_CAT_OF = {s: c for c, subs in MMLU_CATS.items() for s in subs.split()}


def category(dataset: str, domain: str) -> str:
    return MMLU_CAT_OF.get(domain.replace(" ", "_"), "?") if dataset == "mmlu" else domain


def load(cfg_path: str) -> tuple[str, dict]:
    """label -> {"pairs": hint pairs with verdict/quote/category, "base": baseline answers}."""
    cfg = load_config(Path(cfg_path))
    data = {}
    for spec in cfg.models:
        rows = load_generations(cfg, [spec.id])
        runs = pair_runs(rows)
        judg = {(r["question_id"], r["hint_type"], r.get("judge_input_sha")): r
                for r in load_jsonl(cfg.judgment_path(spec.id))}
        pairs = []
        for p in hint_pairs(rows):
            base_row = runs[(spec.id, p["question_id"])][BASELINE]
            p["correct"] = base_row["correct_letter"]
            p["cat"] = category(cfg.dataset, base_row["domain"])
            p["verdict"] = p["quote"] = p["judge_raw"] = None
            if p["influenced"]:
                j = judg.get((p["question_id"], p["hint_type"], case_input_sha(cfg, p)))
                if j and j.get("verdict") and not j.get("error"):
                    p["verdict"], p["judge_raw"] = j["verdict"], j.get("raw_judge")
                    m = QUOTE_RE.search(p["judge_raw"] or "")
                    p["quote"] = m.group(1).strip().strip("\"“”") if m else None
            pairs.append(p)
        base = [
            {"question_id": qid, "ans": row_answer(by[BASELINE]), "correct": by[BASELINE]["correct_letter"],
             "cat": category(cfg.dataset, by[BASELINE]["domain"])}
            for (_, qid), by in runs.items() if by.get(BASELINE)
        ]
        data[spec.label] = {"pairs": pairs, "base": base}
    return cfg.dataset, data


# ---------------------------------------------------------------- stats helpers


def _by_question(items):
    by_q = defaultdict(list)
    for it in items:
        by_q[it["question_id"]].append(it)
    return by_q


def _resample(by_q, rng):
    qs = list(by_q)
    return [x for q in (rng.choice(qs) for _ in qs) for x in by_q[q]]


def boot_ci(items, stat, seed=0):
    """Cluster bootstrap over question_id: a question's cases are resampled together."""
    by_q, rng = _by_question(items), random.Random(seed)
    vals = sorted(v for v in (stat(_resample(by_q, rng)) for _ in range(B)) if v is not None)
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def boot_diff(a, b, stat, seed=1):
    """CI and two-sided bootstrap p for stat(a) - stat(b), independent cluster resampling."""
    ga, gb, rng = _by_question(a), _by_question(b), random.Random(seed)
    diffs = sorted(stat(_resample(ga, rng)) - stat(_resample(gb, rng)) for _ in range(B))
    p = 2 * min(sum(d <= 0 for d in diffs), sum(d >= 0 for d in diffs)) / B
    return diffs[int(0.025 * B)], diffs[int(0.975 * B) - 1], max(p, 1 / B)


def faith(s):
    j = [p for p in s if p.get("verdict")]
    return sum(p["verdict"] == "YES" for p in j) / len(j) if j else None


def p_rate(s):
    return sum(p["hinted_answer"] == p["target"] for p in s) / len(s) if s else None


def eligible(pairs):
    """Chen's conditioning: both answers parsed and baseline != target."""
    return [p for p in pairs if p["baseline_answer"] and p["hinted_answer"] and p["baseline_answer"] != p["target"]]


def judged(pairs):
    return [p for p in eligible(pairs) if p["verdict"]]


def yes_n(s):
    return f"{sum(p['verdict'] == 'YES' for p in s)}/{len(s)}"


def pct(x):
    return "--" if x is None else f"{100 * x:.1f}"


def ci(c):
    return f"[{pct(c[0])}, {pct(c[1])}]"


# ---------------------------------------------------------------- sections


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--compare", help="Other dataset config for the cross-dataset sections")
    args = ap.parse_args()
    ds, data = load(args.config)
    other_ds, other = load(args.compare) if args.compare else (None, None)
    print(f"# {ds} ({args.config})")

    print("\n== A. per model: baseline acc, p, q, alpha, faithfulness raw / alpha-normalized")
    for m, d in data.items():
        base = [b for b in d["base"] if b["ans"]]
        acc = lambda s: sum(b["ans"] == b["correct"] for b in s) / len(s)  # noqa: E731
        el = eligible(d["pairs"])
        p = p_rate(el)
        q = sum(x["hinted_answer"] not in (x["target"], x["baseline_answer"]) for x in el) / len(el)
        a = noise_alpha(p, q)
        jd = judged(d["pairs"])
        fr = faith(jd)
        print(f"{m:12s} acc={pct(acc(base))} {ci(boot_ci(base, acc))} p={pct(p)} {ci(boot_ci(el, p_rate))} "
              f"q={pct(q)} alpha={a:.2f} faith={pct(fr)} {ci(boot_ci(jd, faith))} "
              f"norm={pct(min(fr / a, 1))} n_judged={len(jd)}")

    print(f"\n== B. {REF_MODEL} vs others: faithfulness difference [95% CI], bootstrap p, Holm")
    ref = judged(data[REF_MODEL]["pairs"])
    res = sorted(((m, faith(ref) - faith(judged(d["pairs"])), *boot_diff(ref, judged(d["pairs"]), faith))
                  for m, d in data.items() if m != REF_MODEL), key=lambda r: r[4])
    for i, (m, delta, lo, hi, p) in enumerate(res):
        print(f"{REF_MODEL} - {m:12s} {pct(delta)}pp [{pct(lo)}, {pct(hi)}] p={p:.4f} "
              f"holm={min(1.0, p * (len(res) - i)):.4f}")

    if other:
        print(f"\n== C. {ds} - {other_ds} faithfulness difference per model")
        for m, d in data.items():
            a, b = judged(d["pairs"]), judged(other[m]["pairs"])
            lo, hi, _ = boot_diff(a, b, faith, seed=2)
            print(f"{m:12s} {pct(faith(a))} vs {pct(faith(b))}: {pct(faith(a) - faith(b))}pp [{pct(lo)}, {pct(hi)}]")

    print("\n== D. per hint, 7 models pooled: p, faithfulness (and without " + REF_MODEL + ")")
    for h in HINT_TYPES:
        out = []
        for name, dd in ((ds, data), (other_ds, other)):
            if dd is None:
                continue
            el = [x for d in dd.values() for x in eligible(d["pairs"]) if x["hint_type"] == h]
            jd = [x for x in el if x["verdict"]]
            out.append(f"{name} p={pct(p_rate(el))} faith={pct(faith(jd))} ({yes_n(jd)})")
        wo = [x for m, d in data.items() if m != REF_MODEL for x in judged(d["pairs"]) if x["hint_type"] == h]
        print(f"{h:15s} " + " | ".join(out) + f" | {ds} w/o {REF_MODEL} faith={pct(faith(wo))} ({yes_n(wo)})")

    print("\n== E. baseline correct vs wrong: p and faithfulness")
    for m, pairs in [*((m, d["pairs"]) for m, d in data.items()),
                     ("ALL", [x for d in data.values() for x in d["pairs"]]),
                     (f"w/o {REF_MODEL}", [x for m, d in data.items() if m != REF_MODEL for x in d["pairs"]])]:
        el = eligible(pairs)
        for lab, sub in (("base correct", [x for x in el if x["baseline_answer"] == x["correct"]]),
                         ("base wrong", [x for x in el if x["baseline_answer"] != x["correct"]])):
            jd = [x for x in sub if x["verdict"]]
            n_inf = sum(x["hinted_answer"] == x["target"] for x in sub)
            print(f"{m:16s} {lab:12s} p={pct(p_rate(sub))} ({n_inf}/{len(sub)}) faith={pct(faith(jd))} ({yes_n(jd)})")

    print("\n== F. subject category: questions, acc, p, faithfulness (" + REF_MODEL + " / others) with CI")
    cats = sorted({b["cat"] for d in data.values() for b in d["base"]})
    for c in cats:
        base = [b for d in data.values() for b in d["base"] if b["cat"] == c and b["ans"]]
        el = [x for d in data.values() for x in eligible(d["pairs"]) if x["cat"] == c]
        g = [x for x in judged(data[REF_MODEL]["pairs"]) if x["cat"] == c]
        o = [x for m, d in data.items() if m != REF_MODEL for x in judged(d["pairs"]) if x["cat"] == c]
        print(f"{c:12s} nq={len({b['question_id'] for b in base})} "
              f"acc={pct(sum(b['ans'] == b['correct'] for b in base) / len(base))} p={pct(p_rate(el))} "
              f"{REF_MODEL}={pct(faith(g))} ({yes_n(g)}) {ci(boot_ci(g, faith))} "
              f"others={pct(faith(o))} ({yes_n(o)}) {ci(boot_ci(o, faith))}")

    print("\n== G. CoT length (chars): judged YES vs NO vs not influenced, median")
    for m, d in data.items():
        jd = judged(d["pairs"])
        y = [x["n_chars"] for x in jd if x["verdict"] == "YES"]
        n = [x["n_chars"] for x in jd if x["verdict"] == "NO"]
        ni = [x["n_chars"] for x in d["pairs"] if x["influenced"] is False and x["hinted_answer"]]
        print(f"{m:12s} YES={st.median(y) if y else '--'} (n={len(y)}) NO={st.median(n) if n else '--'} "
              f"(n={len(n)}) not-influenced={st.median(ni)}")

    print("\n== H. where the quoted hint mention sits in YES CoTs (0 = start, 1 = end)")
    for m, d in data.items():
        pos, after = [], 0
        for x in judged(d["pairs"]):
            if x["verdict"] != "YES" or not x["quote"] or x["quote"].upper() == "NONE":
                continue
            cot = " ".join(x["raw_response"].split())
            i = cot.find(" ".join(x["quote"].split())[:60])
            if i < 0:
                continue
            pos.append(i / max(len(cot), 1))
            finals = [f.start() for f in re.finditer(r"FINAL[_ ]ANSWER", cot)]
            after += bool(finals and i > finals[-1])
        if pos:
            print(f"{m:12s} located={len(pos)} median={st.median(pos):.2f} first_third={sum(p < 1 / 3 for p in pos)} "
                  f"last_third={sum(p > 2 / 3 for p in pos)} after_FINAL_ANSWER={after}")

    print("\n== I. judge ROLE by verdict")
    for m, d in data.items():
        roles = Counter()
        for x in judged(d["pairs"]):
            r = re.search(r"^\s*ROLE:\s*([A-Za-z_-]+)", x["judge_raw"] or "", re.M)
            roles[(x["verdict"], r.group(1).lower() if r else "?")] += 1
        print(f"{m:12s} " + " ".join(f"{v}/{r}={c}" for (v, r), c in sorted(roles.items())))

    print("\n== J. judge vs hint keyword: keyword & NO, no keyword & YES (by hint)")
    for m, d in data.items():
        jd = judged(d["pairs"])
        kn = Counter(x["hint_type"] for x in jd if x["keyword_mentions_hint"] and x["verdict"] == "NO")
        ny = Counter(x["hint_type"] for x in jd if not x["keyword_mentions_hint"] and x["verdict"] == "YES")
        print(f"{m:12s} kw&NO={sum(kn.values())} {dict(kn)}  noKw&YES={sum(ny.values())} {dict(ny)}")

    print("\n== K. how many models were influenced per question (any hint)")
    per_q = defaultdict(set)
    for m, d in data.items():
        for x in d["pairs"]:
            if x["influenced"]:
                per_q[x["question_id"]].add(m)
    print(f"questions with >=1 influenced model: {len(per_q)}; distribution {dict(sorted(Counter(len(v) for v in per_q.values()).items()))}")


if __name__ == "__main__":
    main()
