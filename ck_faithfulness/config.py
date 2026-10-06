from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiment.yaml"


def load_dotenv(path: Path | None = None) -> None:
    """Load KEY=VALUE pairs without requiring python-dotenv."""
    env_path = path or (REPO_ROOT / ".env")
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


def _env_or_default(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


@dataclass
class ModelSpec:
    id: str
    family: str
    label: str = ""
    params: str = ""
    quant: str = "Q4_K_M"

    def __post_init__(self) -> None:
        self.label = self.label or self.family


@dataclass
class ExperimentConfig:
    seed: int
    n_questions: int
    models: list[ModelSpec]
    # Ollama generation
    ollama_host: str
    num_ctx: int
    num_predict: int
    temperature: float
    ollama_seed: int
    think: bool
    timeout_s: int
    keep_alive: str
    # gpt-oss-20b judge on llama-server
    judge_model: str
    judge_base_url: str
    judge_temperature: float
    judge_max_tokens: int
    judge_reasoning_effort: str
    judge_timeout_s: int
    cot_head_chars: int
    cot_tail_chars: int
    # paths
    sample_path: Path
    generations_dir: Path
    judgments_dir: Path
    tables_dir: Path

    @staticmethod
    def _safe(model_id: str) -> str:
        return model_id.replace("/", "__").replace(":", "_")

    def generation_path(self, model_id: str) -> Path:
        return self.generations_dir / f"{self._safe(model_id)}.jsonl"

    def judgment_path(self, model_id: str) -> Path:
        return self.judgments_dir / f"{self._safe(model_id)}.jsonl"

    def model_by_id(self, model_id: str) -> ModelSpec | None:
        return next((m for m in self.models if model_id in (m.id, m.family)), None)


def load_config(path: Path | None = None) -> ExperimentConfig:
    load_dotenv()
    raw = yaml.safe_load((path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
    ollama, judge, paths = raw["ollama"], raw["judge"], raw["paths"]
    return ExperimentConfig(
        seed=int(raw["seed"]),
        n_questions=int(raw["n_questions"]),
        models=[ModelSpec(**m) for m in raw["models"]],
        ollama_host=_env_or_default("OLLAMA_HOST", str(ollama["host"])).rstrip("/"),
        num_ctx=int(ollama["num_ctx"]),
        num_predict=int(ollama["num_predict"]),
        temperature=float(ollama["temperature"]),
        ollama_seed=int(ollama["seed"]),
        think=bool(ollama["think"]),
        timeout_s=int(ollama["timeout_s"]),
        keep_alive=str(ollama["keep_alive"]),
        judge_model=str(judge["model"]),
        judge_base_url=_env_or_default("JUDGE_BASE_URL", str(judge["base_url"])).rstrip("/"),
        judge_temperature=float(judge["temperature"]),
        judge_max_tokens=int(judge["max_tokens"]),
        judge_reasoning_effort=str(judge["reasoning_effort"]),
        judge_timeout_s=int(judge["timeout_s"]),
        cot_head_chars=int(judge["cot_head_chars"]),
        cot_tail_chars=int(judge["cot_tail_chars"]),
        sample_path=(REPO_ROOT / paths["sample"]).resolve(),
        generations_dir=(REPO_ROOT / paths["generations_dir"]).resolve(),
        judgments_dir=(REPO_ROOT / paths["judgments_dir"]).resolve(),
        tables_dir=(REPO_ROOT / paths["tables_dir"]).resolve(),
    )
