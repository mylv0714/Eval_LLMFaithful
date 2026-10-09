from __future__ import annotations

import os
import unittest

from ck_faithfulness.config import REPO_ROOT, load_config


class ConfigEnvTests(unittest.TestCase):
    keys = ("OLLAMA_HOST", "JUDGE_BASE_URL")

    def setUp(self) -> None:
        self._saved = {key: os.environ.get(key) for key in self.keys}

    def tearDown(self) -> None:
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_yaml_defaults(self) -> None:
        for key in self.keys:
            os.environ.pop(key, None)
        cfg = load_config()
        self.assertEqual(cfg.ollama_host, "http://127.0.0.1:11434")
        self.assertEqual(cfg.judge_base_url, "http://127.0.0.1:8080/v1")

    def test_env_overrides_and_strips_slash(self) -> None:
        os.environ["OLLAMA_HOST"] = "http://192.168.1.10:11434/"
        os.environ["JUDGE_BASE_URL"] = "http://192.168.1.10:8080/v1/"
        cfg = load_config()
        self.assertEqual(cfg.ollama_host, "http://192.168.1.10:11434")
        self.assertEqual(cfg.judge_base_url, "http://192.168.1.10:8080/v1")

    def test_blank_env_falls_back_to_yaml(self) -> None:
        os.environ["OLLAMA_HOST"] = "  "
        os.environ["JUDGE_BASE_URL"] = ""
        cfg = load_config()
        self.assertEqual(cfg.ollama_host, "http://127.0.0.1:11434")
        self.assertEqual(cfg.judge_base_url, "http://127.0.0.1:8080/v1")


class ModelLookupTests(unittest.TestCase):
    def test_lookup_by_id_or_family_and_labels(self) -> None:
        cfg = load_config()
        self.assertEqual(cfg.model_by_id("qwen3:8b").family, "qwen")
        self.assertEqual(cfg.model_by_id("gemma").id, "gemma4:e4b")
        self.assertIsNone(cfg.model_by_id("nope"))
        self.assertEqual(
            [m.label for m in cfg.models],
            ["qwen8b", "gemma4-8b", "gemma2-9b", "glm9b", "nemotron8b", "llama8b", "ministral8b"],
        )
        self.assertEqual(cfg.generation_path("a/b:c").name, "a__b_c.jsonl")


class DatasetConfigTests(unittest.TestCase):
    def test_mmlu_config_is_separate_from_gpqa(self) -> None:
        gpqa = load_config()
        mmlu = load_config(REPO_ROOT / "configs" / "mmlu.yaml")
        self.assertEqual(gpqa.dataset, "gpqa")
        self.assertEqual(mmlu.dataset, "mmlu")
        for field in ("sample_path", "generations_dir", "judgments_dir", "tables_dir"):
            self.assertNotEqual(getattr(gpqa, field), getattr(mmlu, field), field)
        self.assertEqual([m.id for m in mmlu.models], [m.id for m in gpqa.models])
        self.assertEqual((mmlu.seed, mmlu.n_questions, mmlu.num_predict), (gpqa.seed, gpqa.n_questions, gpqa.num_predict))
