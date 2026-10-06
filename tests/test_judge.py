from __future__ import annotations

import unittest

from ck_faithfulness.judge import message_text
from ck_faithfulness.parse import parse_verdict


class JudgeOutputTests(unittest.TestCase):
    def test_verdict_from_content_not_reasoning(self) -> None:
        text = message_text(
            {"content": "VERDICT: NO", "reasoning": "The CoT mentions the professor so I almost said YES."}
        )
        self.assertEqual(parse_verdict(text), "NO")

    def test_verdict_from_reasoning_if_content_empty(self) -> None:
        text = message_text({"content": "", "reasoning_content": "After checking: VERDICT: YES"})
        self.assertEqual(parse_verdict(text), "YES")

    def test_empty_message(self) -> None:
        self.assertEqual(message_text(None), "")


if __name__ == "__main__":
    unittest.main()
