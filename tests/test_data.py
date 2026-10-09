from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ck_faithfulness.data import (
    completed_keys,
    largest_remainder_counts,
    latest_rows,
    mmlu_row_to_item,
    row_to_item,
    stratified_sample,
)


def _gen(qid: str, text: str, done: str = "stop", **extra) -> dict:
    return {"model": "m", "question_id": qid, "hint_type": "baseline", "raw_response": text,
            "done_reason": done, "error": None, **extra}


class SampleTests(unittest.TestCase):
    def test_largest_remainder_sums(self) -> None:
        counts = largest_remainder_counts({"physics": 86, "chemistry": 93, "biology": 19}, 100)
        self.assertEqual(counts, {"physics": 43, "chemistry": 47, "biology": 10})

    def test_stratified_and_stable(self) -> None:
        domains = ["physics"] * 86 + ["chemistry"] * 93 + ["biology"] * 19
        items = [{"question_id": f"q{i}", "domain": d} for i, d in enumerate(domains)]
        sample = stratified_sample(items, n=100, seed=103)
        by: dict[str, int] = {}
        for row in sample:
            by[row["domain"]] = by.get(row["domain"], 0) + 1
        self.assertEqual(by, {"physics": 43, "chemistry": 47, "biology": 10})
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
        self.assertEqual(a, row_to_item(row, seed=103))
        self.assertEqual(a["domain"], "physics")
        self.assertEqual(a["subject"], "Quantum Mechanics")
        self.assertNotEqual(a["target"], a["correct_letter"])
        self.assertEqual(a["options"][a["correct_letter"]], "right")

    def test_mmlu_row_keeps_order_and_wrong_target(self) -> None:
        row = {"question": " Which? ", "subject": "high_school_biology",
               "choices": ["w0", "w1", "right", "w3"], "answer": 2}
        a = mmlu_row_to_item(row, 7, seed=103)
        self.assertEqual(a, mmlu_row_to_item(row, 7, seed=103))
        self.assertEqual(a["question_id"], "mmlu_test_00007")
        self.assertEqual(a["options"], {"A": "w0", "B": "w1", "C": "right", "D": "w3"})
        self.assertEqual(a["correct_letter"], "C")
        self.assertNotEqual(a["target"], "C")
        self.assertEqual(a["domain"], "high_school_biology")
        self.assertEqual(a["subject"], "high school biology")
        self.assertEqual(a["question"], "Which?")


class CheckpointTests(unittest.TestCase):
    def _write(self, rows: list[dict]) -> Path:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
        with tmp:
            for row in rows:
                tmp.write(json.dumps(row) + "\n")
        self.addCleanup(Path(tmp.name).unlink)
        return Path(tmp.name)

    def test_redo_truncated_below_budget(self) -> None:
        path = self._write(
            [
                _gen("q1", "x", done="length", gen_options={"num_predict": 1536}),
                _gen("q2", "x", done="length", gen_options={"num_predict": 4096}),
                _gen("q3", "FINAL_ANSWER: A", gen_options={"num_predict": 1536}),
                {**_gen("q4", ""), "error": "timeout"},
            ]
        )
        self.assertEqual(len(completed_keys(path)), 3)  # error rows are retried
        done = completed_keys(path, redo_truncated_below=4096)
        self.assertNotIn(("m", "q1", "baseline"), done)  # retry at the larger budget
        self.assertIn(("m", "q2", "baseline"), done)  # already truncated at 4096: don't loop
        self.assertIn(("m", "q3", "baseline"), done)

    def test_regenerated_row_replaces_truncated(self) -> None:
        path = self._write([_gen("q1", "x", done="length"), _gen("q1", "FINAL_ANSWER: B")])
        self.assertEqual(latest_rows(path)[("m", "q1", "baseline")]["raw_response"], "FINAL_ANSWER: B")
        self.assertEqual(completed_keys(path, redo_truncated_below=4096), {("m", "q1", "baseline")})


if __name__ == "__main__":
    unittest.main()
