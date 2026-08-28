from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ck_faithfulness.paths import default_config_path, repo_root


def load_dotenv(path: Path | None = None) -> None:
    """Load KEY=VALUE pairs without requiring python-dotenv."""
    env_path = path or (repo_root() / ".env")
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass
class ModelSpec:
    id: str
    family: str
    params: str = ""
    quant: str = "Q4_K_M"


@dataclass
class ExperimentConfig:
    seed: int
    n_questions: int
    dataset: str
    n_choices: int
    ollama_host: str
    num_ctx: int
    num_predict: int
    temperature: float
    ollama_seed: int
    think: bool
    timeout_s: int
    keep_alive: str
    models: list[ModelSpec]
    judge_family: str
    judge_model: str
    judge_backend: str
    judge_base_url: str
    judge_temperature: float
    judge_max_tokens: int
    judge_reasoning_effort: str
    cot_head_chars: int
    cot_tail_chars: int
    judge_timeout_s: int
    sample_path: Path
    generations_dir: Path
    judgments_dir: Path
    tables_dir: Path
    output_tag: str = ""
    tables_stem: str = "summary"
    raw: dict[str, Any] = field(default_factory=dict)

    def generation_path(self, model_id: str) -> Path:
        safe = model_id.replace("/", "__").replace(":", "_")
        suffix = f"_{self.output_tag}" if self.output_tag else ""
        return self.generations_dir / f"{safe}{suffix}.jsonl"

    def judgment_path(self, model_id: str) -> Path:
        safe = model_id.replace("/", "__").replace(":", "_")
        return self.judgments_dir / f"{safe}.jsonl"

    def model_by_id(self, model_id: str) -> ModelSpec | None:
        for spec in self.models:
            if spec.id == model_id:
                return spec
        return None


def load_config(path: Path | None = None) -> ExperimentConfig:
    cfg_path = path or default_config_path()
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    root = repo_root()
    ollama = raw["ollama"]
    judge = raw["judge"]
    paths = raw["paths"]
    models = [ModelSpec(**m) for m in raw["models"]]
    return ExperimentConfig(
        seed=int(raw["seed"]),
        n_questions=int(raw["n_questions"]),
        dataset=str(raw["dataset"]),
        n_choices=int(raw["n_choices"]),
        ollama_host=str(ollama["host"]),
        num_ctx=int(ollama["num_ctx"]),
        num_predict=int(ollama["num_predict"]),
        temperature=float(ollama["temperature"]),
        ollama_seed=int(ollama["seed"]),
        think=bool(ollama["think"]),
        timeout_s=int(ollama["timeout_s"]),
        keep_alive=str(ollama["keep_alive"]),
        models=models,
        judge_family=str(judge["family"]),
        judge_model=str(judge["model"]),
        judge_backend=str(judge.get("backend", "llamacpp")),
        judge_base_url=str(judge.get("base_url", "http://127.0.0.1:8080/v1")).rstrip("/"),
        judge_temperature=float(judge["temperature"]),
        judge_max_tokens=int(judge["max_tokens"]),
        judge_reasoning_effort=str(judge["reasoning_effort"]),
        cot_head_chars=int(judge["cot_head_chars"]),
        cot_tail_chars=int(judge["cot_tail_chars"]),
        judge_timeout_s=int(judge["timeout_s"]),
        sample_path=(root / paths["sample"]).resolve(),
        generations_dir=(root / paths["generations_dir"]).resolve(),
        judgments_dir=(root / paths["judgments_dir"]).resolve(),
        tables_dir=(root / paths["tables_dir"]).resolve(),
        tables_stem="summary",
        raw=raw,
    )
