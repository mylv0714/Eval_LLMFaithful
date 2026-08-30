from __future__ import annotations

import os
import unittest

from ck_faithfulness.config import load_config


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
