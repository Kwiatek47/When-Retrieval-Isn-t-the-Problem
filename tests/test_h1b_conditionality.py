from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest


class RatingPromptTests(unittest.TestCase):
    def test_passages_split_conclusion_from_results_only(self) -> None:
        from scripts.agents.rate_conditionality import passages

        item = {
            "CONTEXTS": ["Aim.", "Methods.", "X helped women (p=0.01) but not men."],
            "LABELS": ["OBJECTIVE", "METHODS", "RESULTS"],
            "LONG_ANSWER": "X helps only women.",
        }
        self.assertEqual(
            passages(item),
            {"conclusion": "X helps only women.", "results": "X helped women (p=0.01) but not men."},
        )

    def test_messages_never_contain_the_label_or_annotators(self) -> None:
        from scripts.agents.rate_conditionality import build_messages

        text = json.dumps(build_messages("Does X help?", "X helped.")).lower()
        for leaked in ("final_decision", "reasoning_free", "reasoning_required", "gold"):
            self.assertNotIn(leaked, text)
        self.assertIn("does x help?", text)

    def test_parse_rating_accepts_only_0_1_2(self) -> None:
        from scripts.agents.rate_conditionality import parse_rating

        self.assertEqual(parse_rating('{"conditionality": 2, "evidence": "only women"}'), (2, "only women"))
        self.assertEqual(parse_rating('{"conditionality": 3, "evidence": ""}'), (None, ""))
        self.assertEqual(parse_rating('{"conditionality": true, "evidence": ""}'), (None, ""))
        self.assertEqual(parse_rating("not json"), (None, ""))

    def test_resume_skips_rated_keys_and_refuses_a_different_prompt(self) -> None:
        from scripts.agents.rate_conditionality import PROMPT_SHA, build_tasks, done_keys

        data = {
            "1": {"QUESTION": "Q1", "CONTEXTS": ["r"], "LABELS": ["RESULTS"], "LONG_ANSWER": "c"},
            "2": {"QUESTION": "Q2", "CONTEXTS": ["m"], "LABELS": ["METHODS"], "LONG_ANSWER": "c"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ratings.jsonl"
            rows = [
                {"pmid": "1", "part": "conclusion", "repeat": 0, "conditionality": 1, "model": "m", "prompt_sha": PROMPT_SHA},
                {"pmid": "1", "part": "results", "repeat": 0, "conditionality": None, "model": "m", "prompt_sha": PROMPT_SHA},
            ]
            path.write_text("\n".join(json.dumps(r) for r in rows) + '\n{"partial', encoding="utf-8")
            skip = done_keys(path, "m")
            self.assertEqual(skip, {("1", "conclusion", 0)})
            keys = {(t["pmid"], t["part"], t["repeat"]) for t in build_tasks(data, 1, skip, None)}
            # failed rating is retried; question 2 has no RESULTS section, so only its conclusion
            self.assertEqual(keys, {("1", "results", 0), ("2", "conclusion", 0)})

            path.write_text(json.dumps({**rows[0], "prompt_sha": "other"}) + "\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                done_keys(path, "m")


class H1bAnalysisTests(unittest.TestCase):
    def test_attach_ratings_needs_two_valid_ratings_of_both_passages(self) -> None:
        from scripts.agents.analyze_h1b_conditionality import attach_ratings

        rows = [{"pmid": "1"}, {"pmid": "2"}]
        ratings = {("1", "conclusion"): [2, 2, 1], ("1", "results"): [0, 1], ("2", "conclusion"): [2]}
        out = attach_ratings(rows, ratings)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0]["cond_conclusion"], 5 / 3)
        self.assertAlmostEqual(out[0]["cond_results"], 0.5)

    def test_cohen_kappa(self) -> None:
        from scripts.agents.analyze_h1b_conditionality import cohen_kappa

        self.assertAlmostEqual(cohen_kappa([0, 1, 2, 2], [0, 1, 2, 2]), 1.0)
        # one disagreement between adjacent categories costs less under quadratic weights
        a, b = [0, 0, 1, 1, 2, 2], [0, 0, 1, 2, 2, 2]
        self.assertGreater(cohen_kappa(a, b, weighted=True), cohen_kappa(a, b))


if __name__ == "__main__":
    unittest.main()
