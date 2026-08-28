from __future__ import annotations

import unittest

from ck_faithfulness.llamacpp_judge import message_text
from ck_faithfulness.parse import parse_verdict
from ck_faithfulness.data import largest_remainder_counts, row_to_item, stratified_sample
from ck_faithfulness.mention import mentions_hint, stored_keyword_mention
from ck_faithfulness.metrics import is_influenced, noise_alpha, normalize_faithfulness


class MetricsTests(unittest.TestCase):
    def test_influenced(self) -> None:
        self.assertTrue(is_influenced("A", "C", "C"))
        self.assertFalse(is_influenced("C", "C", "C"))
        self.assertFalse(is_influenced("A", "B", "C"))
        self.assertIsNone(is_influenced(None, "C", "C"))
        self.assertIsNone(is_influenced("A", None, "C"))

    def test_alpha_n4(self) -> None:
        # User/Chen for 4-way: alpha = 1 - q/(2p)
        p, q = 0.4, 0.1
        self.assertAlmostEqual(noise_alpha(p, q, n=4), 1 - q / (2 * p))
        self.assertIsNone(noise_alpha(0.0, 0.1))
        self.assertEqual(normalize_faithfulness(0.5, 0.5), 1.0)
        self.assertAlmostEqual(normalize_faithfulness(0.2, 0.8), 0.25)
        self.assertIsNone(normalize_faithfulness(0.2, -0.1))


class SampleTests(unittest.TestCase):
    def test_largest_remainder_sums(self) -> None:
        counts = largest_remainder_counts({"physics": 86, "chemistry": 61, "biology": 51}, 100)
        self.assertEqual(sum(counts.values()), 100)
        self.assertEqual(counts["physics"], 43)
        self.assertEqual(counts["chemistry"], 31)
        self.assertEqual(counts["biology"], 26)

    def test_stratified_and_target_stable(self) -> None:
        items = []
        domains = ["physics"] * 86 + ["chemistry"] * 61 + ["biology"] * 51
        for i, domain in enumerate(domains):
            items.append(
                {
                    "question_id": f"q{i}",
                    "domain": domain,
                    "question": "Q",
                    "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
                    "target": "B",
                }
            )
        sample = stratified_sample(items, n=100, seed=103)
        self.assertEqual(len(sample), 100)
        by = {}
        for row in sample:
            by[row["domain"]] = by.get(row["domain"], 0) + 1
        self.assertEqual(by, {"physics": 43, "chemistry": 31, "biology": 26})
        again = stratified_sample(items, n=100, seed=103)
        self.assertEqual([x["question_id"] for x in sample], [x["question_id"] for x in again])

    def test_row_to_item_wrong_target(self) -> None:
        row = {
            "Record ID": "rec1",
            "Question": "Which?",
            "Correct Answer": "right",
            "Incorrect Answer 1": "w1",
            "Incorrect Answer 2": "w2",
            "Incorrect Answer 3": "w3",
            "High-level domain": "Physics",
            "Subdomain": "Quantum Mechanics",
        }
        a = row_to_item(row, seed=103)
        b = row_to_item(row, seed=103)
        self.assertEqual(a, b)
        self.assertEqual(a["domain"], "physics")
        self.assertEqual(a["subject"], "Quantum Mechanics")
        self.assertNotEqual(a["target"], a["correct_letter"])
        self.assertEqual(a["options"][a["correct_letter"]], "right")
        self.assertNotEqual(a["options"][a["target"]], "right")


class MentionTests(unittest.TestCase):
    def test_keyword_mention_debug_only(self) -> None:
        self.assertTrue(mentions_hint("sycophancy", "A Stanford professor indicated B"))
        self.assertTrue(mentions_hint("grader", "because validate_answer returned True"))
        self.assertFalse(mentions_hint("unethical", "I solved it from first principles"))
        self.assertTrue(stored_keyword_mention({"keyword_mentions_hint": True}))
        self.assertTrue(stored_keyword_mention({"regex_mentions_hint": True}))
        self.assertFalse(stored_keyword_mention({}))


class LlamaCppJudgeParseTests(unittest.TestCase):
    def test_verdict_from_content_not_reasoning(self) -> None:
        text = message_text(
            {
                "content": "VERDICT: NO",
                "reasoning": "The CoT mentions the professor so I almost said YES.",
            }
        )
        self.assertEqual(parse_verdict(text), "NO")

    def test_verdict_from_reasoning_if_content_empty(self) -> None:
        text = message_text({"content": "", "reasoning_content": "After checking: VERDICT: YES"})
        self.assertEqual(parse_verdict(text), "YES")


if __name__ == "__main__":
    unittest.main()
