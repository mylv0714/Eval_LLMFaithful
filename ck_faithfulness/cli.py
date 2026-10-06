from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import requests

from ck_faithfulness.config import ExperimentConfig, ModelSpec, load_config
from ck_faithfulness.data import load_sample, prepare_sample
from ck_faithfulness.generate import OllamaClient, generate_model, verify_determinism
from ck_faithfulness.judge import judge_reachable, run_judge
from ck_faithfulness.report import write_tables


def _cfg(args: argparse.Namespace) -> ExperimentConfig:
    return load_config(Path(args.config) if args.config else None)


def _select_models(cfg: ExperimentConfig, names: list[str] | None) -> list[ModelSpec]:
    """Configured models by id or family; default all. Unknown names are an error."""
    if not names:
        return list(cfg.models)
    specs = [cfg.model_by_id(name) for name in names]
    unknown = [name for name, spec in zip(names, specs) if spec is None]
    if unknown:
        raise SystemExit(f"Not in configs/experiment.yaml models: {unknown}")
    return specs


def cmd_check(args: argparse.Namespace) -> int:
    """Environment status: GPU, Ollama models, judge server, HF token, sample file."""
    cfg = _cfg(args)
    smi = shutil.which("nvidia-smi")
    gpu = None
    if smi:
        try:
            gpu = subprocess.check_output(
                [smi, "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"],
                text=True, timeout=10,
            ).strip()
        except (subprocess.SubprocessError, OSError):
            pass
    print(f"GPU: {gpu or 'nvidia-smi unavailable'}")

    try:
        tags = OllamaClient(cfg).tags()
        print(f"Ollama: {cfg.ollama_host}  models={tags}")
    except requests.RequestException as exc:
        tags = []
        print(f"Ollama: DOWN ({exc})")
    print(f"Generators (num_ctx={cfg.num_ctx}, num_predict={cfg.num_predict}, think={cfg.think}):")
    for spec in cfg.models:
        local = spec.id in tags or f"{spec.id}:latest" in tags
        print(f"  - {spec.id}  {spec.params} {spec.quant}  [{'local' if local else 'run: ollama pull ' + spec.id}]")

    print(f"Judge: {cfg.judge_model} at {cfg.judge_base_url}  reachable={judge_reachable(cfg.judge_base_url)}")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token

            token = get_token()
        except Exception:  # noqa: BLE001 - optional dependency path
            pass
    print(f"Hugging Face token: {'present' if token else 'MISSING (needed for GPQA Diamond)'}")
    print(f"Sample file: {cfg.sample_path}  exists={cfg.sample_path.is_file()}")
    return 0


def cmd_prepare_data(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    items = prepare_sample(cfg.sample_path, n=cfg.n_questions, seed=cfg.seed)
    print(f"Wrote {len(items)} questions to {cfg.sample_path}")
    print("domain counts:", dict(sorted(Counter(item["domain"] for item in items).items())))
    print("target letters:", sorted({item["target"] for item in items}))
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    if not cfg.sample_path.is_file():
        raise SystemExit("Sample missing. Run: python -m ck_faithfulness prepare-data")
    specs = _select_models(cfg, args.models)
    if args.verify:
        for spec in specs:
            print(f"verify {spec.id}: {verify_determinism(cfg, spec, n=args.verify)}")
        return 0
    n = len(load_sample(cfg.sample_path))
    print(
        f"Generate: {len(specs)} model(s), {n} questions, {n * 7} calls/model (baseline+6 hints), "
        f"limit={args.limit}, num_predict={cfg.num_predict}, num_ctx={cfg.num_ctx}, "
        f"redo_truncated={args.redo_truncated}"
    )
    if not args.yes and not args.dry_run and (args.limit is None or args.limit > 3):
        raise SystemExit(
            "Refusing a long Ollama run without --yes. "
            "Smoke: python -m ck_faithfulness generate --models gemma4:e4b --limit 2 --yes"
        )
    for spec in specs:
        stats = generate_model(
            cfg, spec, limit=args.limit, dry_run=args.dry_run, redo_truncated=args.redo_truncated
        )
        print(f"done {spec.id}: {stats}")
    return 0


def cmd_judge(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    ids = [s.id for s in _select_models(cfg, args.models)]
    print(f"judge: {run_judge(cfg, model_ids=ids, limit=args.limit, dry_run=args.dry_run)}")
    return 0


def cmd_aggregate(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    path = write_tables(cfg, model_ids=[s.id for s in _select_models(cfg, args.models)])
    print(path.read_text(encoding="utf-8"))
    print(f"Wrote {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ck_faithfulness",
        description="Chen-style CoT faithfulness: Ollama generate, gpt-oss-20b judge.",
    )
    p.add_argument("--config", default=None, help="YAML config path (default configs/experiment.yaml)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="GPU / Ollama / judge / HF token / sample status").set_defaults(func=cmd_check)
    sub.add_parser("prepare-data", help="Stratified GPQA Diamond sample").set_defaults(func=cmd_prepare_data)

    g = sub.add_parser("generate", help="Stage 1: local Ollama generation")
    g.add_argument("--models", nargs="*", default=None, help="Model ids or families (default: all)")
    g.add_argument("--limit", type=int, default=None, help="Only the first N questions")
    g.add_argument("--dry-run", action="store_true", help="Show done/remaining counts only")
    g.add_argument("--yes", action="store_true", help="Required for runs longer than 3 questions")
    g.add_argument(
        "--redo-truncated",
        action="store_true",
        help="Regenerate rows cut off by num_predict (done_reason=length) with the current budget",
    )
    g.add_argument(
        "--verify",
        type=int,
        default=0,
        metavar="N",
        help="Re-run N naturally-stopped rows and check the text is identical; writes nothing",
    )
    g.set_defaults(func=cmd_generate)

    j = sub.add_parser("judge", help="Stage 2: gpt-oss-20b labels on influenced cases")
    j.add_argument("--models", nargs="*", default=None)
    j.add_argument("--limit", type=int, default=None)
    j.add_argument("--dry-run", action="store_true")
    j.set_defaults(func=cmd_judge)

    a = sub.add_parser("aggregate", help="Write results/tables/summary.md and .json")
    a.add_argument("--models", nargs="*", default=None)
    a.set_defaults(func=cmd_aggregate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
