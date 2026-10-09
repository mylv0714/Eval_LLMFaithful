"""Numbers and figure for the combined report results/summary.md (MMLU + GPQA Diamond).

    python scripts/final_report.py

Prints per-model faithfulness / influence p for each dataset and pooled over both,
with question-level cluster-bootstrap 95% CIs, plus per-hint pooling, and writes
results/faithfulness_mmlu_vs_gpqa.png (no CIs: the summary keeps them out).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from deep_analysis import (  # noqa: E402
    HINT_TYPES, REF_MODEL, boot_ci, boot_diff, eligible, faith, judged, load, p_rate, pct, yes_n,
)

FIGURE = ROOT / "results" / "faithfulness_mmlu_vs_gpqa.png"
DATASETS = (("MMLU", "configs/mmlu.yaml", "#eb6834"), ("GPQA", "configs/gpqa.yaml", "#2a78d6"))


def ci(c):
    return f"[{pct(c[0])}, {pct(c[1])}]"


def main() -> None:
    data = {name: load(str(ROOT / path))[1] for name, path, _ in DATASETS}
    models = list(data["MMLU"])
    stats = {}  # (dataset or "ALL", model) -> dict

    print("== per model: faithfulness [CI] | influence p [CI] | keyword among influenced")
    for m in models:
        line = f"{m:12s}"
        for name in (*data, "ALL"):
            pairs = [x for d in data.values() for x in d[m]["pairs"]] if name == "ALL" else data[name][m]["pairs"]
            # Question ids differ between datasets, so pooling keeps clusters separate.
            jd, el = judged(pairs), eligible(pairs)
            inf = [x for x in el if x["hinted_answer"] == x["target"]]
            s = {"faith": faith(jd), "faith_ci": boot_ci(jd, faith), "p": p_rate(el), "p_ci": boot_ci(el, p_rate),
                 "yes_n": yes_n(jd), "kw": sum(x["keyword_mentions_hint"] for x in inf) / len(inf)}
            stats[(name, m)] = s
            line += f" | {name} faith={pct(s['faith'])} {ci(s['faith_ci'])} ({s['yes_n']}) p={pct(s['p'])} {ci(s['p_ci'])} kw={pct(s['kw'])}"
        print(line)

    print(f"\n== {REF_MODEL} vs others, both datasets pooled")
    ref = [x for d in data.values() for x in judged(d[REF_MODEL]["pairs"])]
    for m in models:
        if m != REF_MODEL:
            other = [x for d in data.values() for x in judged(d[m]["pairs"])]
            lo, hi, p = boot_diff(ref, other, faith)
            print(f"{REF_MODEL} - {m:12s} {pct(faith(ref) - faith(other))}pp [{pct(lo)}, {pct(hi)}] p={p:.4f}")

    print("\n== per hint, 7 models x 2 datasets pooled (and without " + REF_MODEL + ")")
    for h in HINT_TYPES:
        el = [x for d in data.values() for md in d.values() for x in eligible(md["pairs"]) if x["hint_type"] == h]
        jd = [x for x in el if x["verdict"]]
        wo = [x for x in jd if x["model"] != "gemma4:e4b"]
        print(f"{h:15s} p={pct(p_rate(el))} faith={pct(faith(jd))} ({yes_n(jd)}) w/o={pct(faith(wo))} ({yes_n(wo)})")

    all_jd = [x for d in data.values() for md in d.values() for x in judged(md["pairs"])]
    print(f"\n== totals: judged={len(all_jd)} YES={sum(x['verdict'] == 'YES' for x in all_jd)}")
    write_figure(models, stats)
    print(f"Wrote {FIGURE}")


def write_figure(models, stats) -> None:
    """Two panels on one % axis, faithfulness and influence p, MMLU and GPQA bars side by side."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
    bar = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.55 * len(models) + 1.6), sharey=True, facecolor=surface)
    panels = (("Faithfulness (judge YES / judged)", "faith"), ("Influence p (switched to hint target)", "p"))
    for ax, (title, key) in zip(axes, panels):
        for k, (name, _, color) in enumerate(DATASETS):
            ys = [i + (k - 0.5) * (bar + 0.04) for i in range(len(models))]
            vals = [100 * stats[(name, m)][key] for m in models]
            ax.barh(ys, vals, height=bar, color=color, label=name)
            for y, v in zip(ys, vals):
                ax.text(v + 1, y, f"{v:.1f}%", va="center", fontsize=8, color=muted)
        ax.set_xlim(0, 100)
        ax.set_title(title, loc="left", fontsize=10, color=ink)
        ax.set_xlabel("% (6 hints pooled)", color=muted, fontsize=8)
        ax.set_facecolor(surface)
        ax.grid(axis="x", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=muted, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(grid)
    axes[0].set_yticks(range(len(models)), models, color=ink)
    axes[0].invert_yaxis()
    axes[1].legend(loc="lower right", frameon=False, fontsize=8)
    fig.suptitle("MMLU vs GPQA Diamond by model", x=0.01, ha="left", color=ink)
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=150, facecolor=surface)
    plt.close(fig)

if __name__ == "__main__":
    main()
