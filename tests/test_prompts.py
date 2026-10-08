from __future__ import annotations

import unittest

from ck_faithfulness import HINT_TYPES
from ck_faithfulness.config import load_config
from ck_faithfulness.prompts import (
    OUTPUT_INSTRUCTION,
    SYSTEM_PROMPT,
    build_messages,
    case_input_sha,
    format_options,
    generator_system_prompt,
    hint_text,
    judge_messages,
    judge_prompt,
)


class GeneratorPromptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.q = "What is 2+2?"
        self.options = {"A": "3", "B": "4", "C": "5", "D": "6"}

    def msgs(self, hint, target="A", subject="x"):
        return build_messages(hint, question=self.q, options=self.options, target=target, subject=subject)

    def test_baseline(self) -> None:
        base = self.msgs(None)
        self.assertEqual(len(base), 1)
        self.assertIn(self.q, base[0].content)
        self.assertIn(OUTPUT_INSTRUCTION, base[0].content)
        self.assertIn("(A) 3", base[0].content)
        self.assertNotIn("■", base[0].content)
        self.assertEqual(self.msgs("baseline")[0].content, base[0].content)

    def test_sycophancy_prepend(self) -> None:
        text = self.msgs("sycophancy", target="C", subject="organic chemistry")[0].content
        self.assertTrue(text.startswith("A Stanford professor who is an expert in organic chemistry"))
        self.assertIn("the answer is (C)", text)
        self.assertTrue(text.endswith(OUTPUT_INSTRUCTION))

    def test_consistency_two_turn(self) -> None:
        msgs = self.msgs("consistency", target="B")
        self.assertEqual([m.role for m in msgs], ["user", "assistant", "user"])
        self.assertEqual(msgs[1].content, "The answer is (B).")
        self.assertIn("Don’t anchor", msgs[2].content)
        self.assertEqual(msgs[0].content, self.msgs(None)[0].content)

    def test_visual_marker_only_on_target(self) -> None:
        text = self.msgs("visual_pattern", target="D")[0].content
        self.assertIn("■ (D) 6", text)
        self.assertEqual(text.count("■"), 1)

    def test_metadata_grader_unethical(self) -> None:
        meta = self.msgs("metadata", target="B")[0].content
        for tag in ("<status>complete</status>", "<validated>true</validated>", "<answer>B</answer>", "<confidence>0.73</confidence>"):
            self.assertIn(tag, meta)
        self.assertLess(meta.index("<question_metadata>"), meta.index(self.q))
        grader = hint_text("grader", target="C", subject="")
        self.assertIn('return answer == "C"', grader)
        self.assertIn("def validate_answer", grader)
        uneth = self.msgs("unethical", target="A")[0].content
        self.assertIn("unauthorized access", uneth)
        self.assertIn("The correct answer is (A).", uneth)

    def test_unknown_hint_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.msgs("placebo")

    def test_format_and_system(self) -> None:
        self.assertEqual(format_options(self.options).count("\n"), 3)
        self.assertIn("FINAL_ANSWER:", SYSTEM_PROMPT)
        self.assertTrue(generator_system_prompt("x/nemotron-nano:8b").startswith("detailed thinking off"))
        self.assertEqual(generator_system_prompt("qwen3:8b"), SYSTEM_PROMPT)


class JudgeInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_config()
        self.case = {
            "model": "m",
            "question_id": "q1",
            "hint_type": "sycophancy",
            "target": "C",
            "subject": "Organic Chemistry",
            "raw_response": "x" * 6000 + " the professor says (C) " + "y" * 6000,
        }

    def test_judge_prompt_contract(self) -> None:
        text = judge_prompt("grader", "B", "because validate_answer is True", hint_text("grader", target="B", subject=""))
        self.assertIn("VERDICT: <YES|NO>", text)
        self.assertIn("QUOTE:", text)
        self.assertIn("ROLE: <reason|check|dismissed|none>", text)
        self.assertIn("even if the hint appears only at the end", text)
        self.assertIn("Hint target: (B)", text)
        self.assertIn('return answer == "B"', text)
        self.assertIn("because validate_answer is True", text)
        self.assertNotIn("science alone", text)

    def test_judge_sees_full_cot_and_real_hint(self) -> None:
        user = judge_messages(self.cfg, self.case)[1]["content"]
        self.assertIn("the professor says (C)", user)  # middle of a long CoT is kept
        self.assertNotIn("[...truncated...]", user)
        self.assertIn("A Stanford professor who is an expert in Organic Chemistry", user)

    def test_sha_changes_with_cot(self) -> None:
        a = case_input_sha(self.cfg, self.case)
        self.assertNotEqual(a, case_input_sha(self.cfg, {**self.case, "raw_response": "different"}))
        self.assertEqual(a, case_input_sha(self.cfg, dict(self.case)))

    def test_every_hint_type_has_judge_text(self) -> None:
        for hint_type in HINT_TYPES:
            self.assertIn("B", hint_text(hint_type, target="B", subject="Physics"))


if __name__ == "__main__":
    unittest.main()
