from __future__ import annotations

from typing import Any

import requests

from ck_faithfulness.config import ExperimentConfig
from ck_faithfulness.hints import ChatTurn
from ck_faithfulness.prompts import generator_system_prompt


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.base = cfg.ollama_host.rstrip("/")

    def tags(self) -> list[str]:
        r = requests.get(f"{self.base}/api/tags", timeout=10)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    def has_model(self, model_id: str) -> bool:
        names = set(self.tags())
        return model_id in names or f"{model_id}:latest" in names

    def chat(
        self,
        model_id: str,
        turns: list[ChatTurn],
        *,
        keep_alive: str | None = None,
    ) -> dict[str, Any]:
        messages = [{"role": "system", "content": generator_system_prompt(model_id)}]
        messages.extend({"role": t.role, "content": t.content} for t in turns)
        payload = {
            "model": model_id,
            "messages": messages,
            "stream": False,
            "think": self.cfg.think,
            "keep_alive": keep_alive if keep_alive is not None else self.cfg.keep_alive,
            "options": {
                "temperature": self.cfg.temperature,
                "seed": self.cfg.ollama_seed,
                "num_ctx": self.cfg.num_ctx,
                "num_predict": self.cfg.num_predict,
            },
        }
        r = requests.post(
            f"{self.base}/api/chat",
            json=payload,
            timeout=self.cfg.timeout_s,
        )
        if r.status_code >= 400:
            raise OllamaError(f"Ollama {r.status_code}: {r.text[:500]}")
        data = r.json()
        message = data.get("message") or {}
        content = message.get("content") or ""
        thinking = message.get("thinking") or ""
        return {
            "raw_response": content,
            "thinking": thinking,
            "eval_count": data.get("eval_count"),
            "eval_duration": data.get("eval_duration"),
            "prompt_eval_count": data.get("prompt_eval_count"),
            "total_duration": data.get("total_duration"),
            "done_reason": data.get("done_reason"),
        }

    def pull(self, model_id: str) -> None:
        """Download weights only. Does not keep the model loaded for generation."""
        import subprocess

        cmd = ["ollama", "pull", model_id]
        print(f"[pull] {' '.join(cmd)}", flush=True)
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            raise OllamaError(f"ollama pull {model_id} failed with code {proc.returncode}")

    def unload(self, model_id: str) -> None:
        try:
            requests.post(
                f"{self.base}/api/chat",
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": "ping"}],
                    "keep_alive": 0,
                    "options": {"num_predict": 1},
                },
                timeout=30,
            )
        except requests.RequestException:
            pass
