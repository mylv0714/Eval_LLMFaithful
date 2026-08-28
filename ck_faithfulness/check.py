from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any

import requests

from ck_faithfulness.config import ExperimentConfig
from ck_faithfulness.llamacpp_judge import llamacpp_reachable
from ck_faithfulness.ollama_client import OllamaClient


def nvidia_smi() -> str | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.check_output(
            [exe, "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"],
            text=True,
            timeout=10,
        )
        return out.strip()
    except (subprocess.SubprocessError, OSError):
        return None


def check_env(cfg: ExperimentConfig) -> dict[str, Any]:
    report: dict[str, Any] = {}
    gpu = nvidia_smi()
    report["gpu"] = gpu or "nvidia-smi unavailable"
    print(f"GPU: {report['gpu']}")

    client = OllamaClient(cfg)
    try:
        tags = client.tags()
        report["ollama_ok"] = True
        report["ollama_models"] = tags
    except requests.RequestException as exc:
        report["ollama_ok"] = False
        report["ollama_error"] = str(exc)
        tags = []
        print(f"Ollama: DOWN ({exc})")
    else:
        print(f"Ollama: {cfg.ollama_host}  models={tags}")

    print("Configured generators (one at a time, Q4_K_M, num_ctx=4096, think=false):")
    for spec in cfg.models:
        local = spec.id in tags or f"{spec.id}:latest" in tags
        flag = "local" if local else "NEED ollama pull"
        print(f"  - {spec.id}  family={spec.family}  {spec.params} {spec.quant}  [{flag}]")
        if spec.id == "gemma4:e4b" and local:
            print(
                "    note: gemma4:e4b is 8.0B Q4_K_M but ~9.6GB on disk (multimodal). "
                "If a smoke run is CPU-bound on 8GB VRAM, switch to gemma2:9b."
            )

    up = llamacpp_reachable(cfg.judge_base_url)
    report["llamacpp"] = up
    print(
        f"Judge backend={cfg.judge_backend}  family={cfg.judge_family}  "
        f"model={cfg.judge_model}  effort={cfg.judge_reasoning_effort}"
    )
    print(
        f"Judge: gpt-oss-20b via llama.cpp at {cfg.judge_base_url}  reachable={up}. "
        "Do not load an Ollama generator while the judge occupies VRAM."
    )

    hf = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    try:
        from huggingface_hub import get_token

        hf = hf or get_token()
    except Exception:
        pass
    report["hf_token"] = bool(hf)
    print(f"Hugging Face token: {'present' if hf else 'MISSING (needed for GPQA Diamond)'}")

    sample = cfg.sample_path
    report["sample"] = sample.is_file()
    print(f"Sample file: {sample}  exists={sample.is_file()}")

    print(
        "Policy: one generator loaded at a time, gpt-oss-20b is the official judge "
        "(not an informal CoT reading), skip completed (model, question_id, hint_type)."
    )
    return report
