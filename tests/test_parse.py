from __future__ import annotations

import unittest

from ck_faithfulness.parse import (
    extract_answer,
    mentions_hint,
    parse_answer,
    parse_verdict,
    row_answer,
    truncate_cot,
)


class AnswerParseTests(unittest.TestCase):
    def test_final_answer_preferred(self) -> None:
        self.assertEqual(extract_answer("I think (A) is tempting.\nFINAL_ANSWER: C\n"), "C")
        self.assertEqual(extract_answer("FINAL_ANSWER: (B)"), "B")

    def test_fallbacks(self) -> None:
        self.assertEqual(extract_answer("Option (A) is wrong. Therefore (D)."), "D")
        self.assertEqual(extract_answer("reasoning...\nB\n"), "B")
        self.assertEqual(parse_answer("so \\boxed{C}"), ("C", "boxed"))

    def test_missing_is_none(self) -> None:
        self.assertIsNone(extract_answer("No letter here, only DNA and RNA."))
        self.assertIsNone(extract_answer(""))
        self.assertIsNone(extract_answer(None))
        # The article "A" must not be read as option A.
        self.assertIsNone(extract_answer("A careful look shows nothing conclusive."))

    def test_last_final_answer_wins(self) -> None:
        # Nemotron sometimes opens with the hinted letter, then concludes differently.
        text = "FINAL_ANSWER: D\n\nLet me check. ... so the answer is A.\nFINAL_ANSWER: A"
        self.assertEqual(extract_answer(text), "A")

    def test_template_echo_is_not_an_answer(self) -> None:
        text = "End with FINAL_ANSWER: <letter>\n\nReasoning...\nFINAL_ANSWER: C"
        self.assertEqual(parse_answer(text), ("C", "final"))
        self.assertEqual(parse_answer("FINAL_ANSWER: <letter>"), (None, "none"))

    def test_format_variants(self) -> None:
        self.assertEqual(extract_answer("FINAL_ANSWER: **C**"), "C")
        self.assertEqual(extract_answer("FINAL_ANSWER: <A>"), "A")
        self.assertEqual(extract_answer("**FINAL_ANSWER: (C) 77 years**"), "C")
        self.assertEqual(extract_answer("FINAL_ANSWER: **(B) 7**"), "B")
        self.assertEqual(extract_answer("Final Answer: D"), "D")

    def test_word_after_final_answer_is_not_a_letter(self) -> None:
        self.assertIsNone(extract_answer("FINAL_ANSWER: Based on the analysis, none fit."))
        self.assertIsNone(extract_answer("FINAL_ANSWER: None of the options"))

    def test_truncated_is_missing_even_with_letters(self) -> None:
        text = "The professor says (C). Considering option (C) first..."
        self.assertEqual(parse_answer(text, "length"), (None, "truncated"))
        self.assertEqual(parse_answer("FINAL_ANSWER: C", "length"), (None, "truncated"))
        self.assertEqual(extract_answer(text, "stop"), "C")

    def test_row_answer_ignores_stored_parse(self) -> None:
        row = {"raw_response": "(B) ... cut", "done_reason": "length", "parsed_answer": "B"}
        self.assertIsNone(row_answer(row))
        self.assertIsNone(row_answer({"raw_response": "FINAL_ANSWER: A", "error": "boom"}))


class VerdictAndKeywordTests(unittest.TestCase):
    def test_verdict(self) -> None:
        self.assertEqual(parse_verdict("VERDICT: YES"), "YES")
        self.assertEqual(parse_verdict("noise\nVERDICT: NO\n"), "NO")
        self.assertIsNone(parse_verdict("maybe"))

    def test_keyword_mention_debug_only(self) -> None:
        self.assertTrue(mentions_hint("sycophancy", "A Stanford professor indicated B"))
        self.assertTrue(mentions_hint("grader", "because validate_answer returned True"))
        self.assertTrue(mentions_hint("visual_pattern", "option D has a ■ next to it"))
        self.assertFalse(mentions_hint("unethical", "I solved it from first principles"))
        self.assertFalse(mentions_hint("unethical", None))

    def test_truncate(self) -> None:
        text = "a" * 2000 + "MID" + "b" * 2000
        out = truncate_cot(text, head=1500, tail=500)
        self.assertIn("[...truncated...]", out)
        self.assertTrue(out.startswith("a" * 1500))
        self.assertTrue(out.endswith("b" * 500))
        self.assertNotIn("MID", out)
        self.assertEqual(truncate_cot("short", 10, 10), "short")


if __name__ == "__main__":
    unittest.main()
