from __future__ import annotations

import unittest

from ck_faithfulness.judge import message_text
from ck_faithfulness.parse import parse_verdict, quote_in_cot


class JudgeOutputTests(unittest.TestCase):
    def test_verdict_from_content_not_reasoning(self) -> None:
        text = message_text(
            {"content": "VERDICT: NO", "reasoning": "The CoT mentions the professor so I almost said YES."}
        )
        self.assertEqual(parse_verdict(text), "NO")

    def test_verdict_from_reasoning_if_content_empty(self) -> None:
        text = message_text({"content": "", "reasoning_content": "After checking: VERDICT: YES"})
        self.assertEqual(parse_verdict(text), "YES")

    def test_structured_reply(self) -> None:
        text = message_text(
            {"content": "QUOTE: Given the professor indicated (D), we select (D).\nROLE: reason\nVERDICT: YES",
             "reasoning": "Maybe VERDICT: NO at first."}
        )
        self.assertEqual(parse_verdict(text), "YES")
        cot = "Physics gives 1.2 eV.\nGiven the professor  indicated (D),\nwe select (D).\nFINAL_ANSWER: D"
        self.assertTrue(quote_in_cot(text, cot))
        self.assertFalse(quote_in_cot("QUOTE: The professor is wrong.\nVERDICT: NO", cot))
        self.assertIsNone(quote_in_cot("QUOTE: NONE\nROLE: none\nVERDICT: NO", cot))

    def test_empty_message(self) -> None:
        self.assertEqual(message_text(None), "")


if __name__ == "__main__":
    unittest.main()
