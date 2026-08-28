from __future__ import annotations

import argparse
from pathlib import Path

from ck_faithfulness.aggregate import write_tables
from ck_faithfulness.check import check_env
from ck_faithfulness.config import ExperimentConfig, load_config, load_dotenv
from ck_faithfulness.data import load_sample, prepare_sample
from ck_faithfulness.generate import generate_model
from ck_faithfulness.judge import run_judge


def _cfg(args: argparse.Namespace) -> ExperimentConfig:
    load_dotenv()
    path = Path(args.config) if getattr(args, "config", None) else None
    return load_config(path)


def _select_models(cfg: ExperimentConfig, names: list[str] | None):
    if not names:
        return list(cfg.models)
    selected = []
    for name in names:
        spec = cfg.model_by_id(name)
        if spec is None:
            spec = next((m for m in cfg.models if m.family == name), None)
        if spec is None:
            from ck_faithfulness.config import ModelSpec

            family = name.split("/")[-1].split(":")[0]
            spec = ModelSpec(id=name, family=family)
        selected.append(spec)
    return selected


def cmd_check(args: argparse.Namespace) -> int:
    check_env(_cfg(args))
    return 0


def cmd_prepare_data(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    items = prepare_sample(cfg.sample_path, n=cfg.n_questions, seed=cfg.seed)
    domains: dict[str, int] = {}
    for item in items:
        domains[item["domain"]] = domains.get(item["domain"], 0) + 1
    print(f"Wrote {len(items)} questions to {cfg.sample_path}")
    print("domain counts:", dict(sorted(domains.items())))
    targets = {item["target"] for item in items}
    print("target letters:", sorted(targets))
    return 0


def cmd_pull(args: argparse.Namespace) -> int:
    """Download missing Ollama models. Avoid pulling while another generator is loaded."""
    from ck_faithfulness.ollama_client import OllamaClient, OllamaError

    cfg = _cfg(args)
    specs = _select_models(cfg, args.models)
    client = OllamaClient(cfg)
    missing = []
    for spec in specs:
        if client.has_model(spec.id):
            print(f"[pull] already local: {spec.id}")
        else:
            missing.append(spec)
    if not missing:
        print("[pull] nothing to download")
        return 0
    print(f"[pull] {len(missing)} model(s) to download. This uses disk/network, not the gpt-oss-20b judge.")
    print("Pull one family at a time. Do not `ollama run` them until the current generator has unloaded.")
    for spec in missing:
        try:
            client.pull(spec.id)
        except OllamaError as exc:
            raise SystemExit(str(exc)) from exc
        if not client.has_model(spec.id):
            raise SystemExit(f"Pull finished but {spec.id} is still missing from `ollama list`.")
        print(f"[pull] ready: {spec.id}")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    if not cfg.sample_path.is_file():
        raise SystemExit("Sample missing. Run: python -m ck_faithfulness prepare-data")
    n = len(load_sample(cfg.sample_path))
    specs = _select_models(cfg, args.models)
    if args.num_predict is not None:
        cfg.num_predict = args.num_predict
        if not args.tag:
            cfg.output_tag = f"np{args.num_predict}"
    if args.tag:
        cfg.output_tag = args.tag
    print(
        f"Generate: {len(specs)} model(s), {n} questions, "
        f"{n * 7} calls/model (baseline+6 hints), limit={args.limit}, "
        f"num_predict={cfg.num_predict}, tag={cfg.output_tag or '-'}"
    )
    if not args.yes and not args.dry_run and (args.limit is None or args.limit > 3):
        raise SystemExit(
            "Refusing a long Ollama run without --yes. "
            "Smoke: python -m ck_faithfulness generate --models gemma4:e4b --limit 2 --yes\n"
            "Full gemma4: python -m ck_faithfulness generate --models gemma4:e4b --yes"
        )
    for spec in specs:
        stats = generate_model(cfg, spec, limit=args.limit, dry_run=args.dry_run)
        print(f"done {spec.id}: {stats}")
    return 0


def cmd_judge(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    stats = run_judge(
        cfg,
        model_ids=args.models,
        limit=args.limit,
        dry_run=args.dry_run,
    )
    print(f"judge: {stats}")
    return 0


def cmd_aggregate(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    path = write_tables(cfg, model_ids=args.models)
    print(f"Wrote {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ck_faithfulness",
        description="Chen-style CoT faithfulness: Ollama generate, gpt-oss-20b judge.",
    )
    p.add_argument("--config", default=None, help="YAML config path")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="GPU / Ollama / API key / sample status")
    c.set_defaults(func=cmd_check)

    d = sub.add_parser("prepare-data", help="Stratified GPQA Diamond sample, seed=103")
    d.set_defaults(func=cmd_prepare_data)

    pl = sub.add_parser("pull", help="ollama pull missing generators (download only)")
    pl.add_argument("--models", nargs="*", default=None, help="Subset of ids/families; default=all missing")
    pl.set_defaults(func=cmd_pull)

    g = sub.add_parser("generate", help="Local Ollama generation (not judging)")
    g.add_argument("--models", nargs="*", default=None, help="Model ids or families")
    g.add_argument("--limit", type=int, default=None, help="Cap number of questions")
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--yes", action="store_true", help="Required for runs with --limit>3 or full")
    g.add_argument("--num-predict", type=int, default=None, help="Override Ollama num_predict; writes a separate jsonl")
    g.add_argument("--tag", default=None, help="Output filename suffix, e.g. np1024")
    g.set_defaults(func=cmd_generate)

    j = sub.add_parser("judge", help="gpt-oss-20b Stage-2 labels on influenced cases only")
    j.add_argument("--models", nargs="*", default=None)
    j.add_argument("--limit", type=int, default=None)
    j.add_argument("--dry-run", action="store_true")
    j.set_defaults(func=cmd_judge)

    a = sub.add_parser("aggregate", help="Influence / faithfulness tables")
    a.add_argument("--models", nargs="*", default=None)
    a.set_defaults(func=cmd_aggregate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
