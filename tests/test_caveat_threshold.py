from __future__ import annotations

import unittest

import numpy as np

from scripts.agents.analyze_caveat_threshold import (
    MODELS,
    PROMPTS,
    analyze,
    choose_threshold,
    conclusion_effect,
    threshold_labels,
)


class ThresholdTests(unittest.TestCase):
    def test_threshold_matches_the_target_share(self) -> None:
        p = np.array([0.0, 0.05, 0.1, 0.1, 0.3, 0.5])
        self.assertEqual(choose_threshold(p, 2 / 6), 0.3)   # p >= 0.3 -> 2 of 6
        self.assertEqual(choose_threshold(p, 3 / 6), 0.3)   # 0.3 gives 2/6, 0.1 gives 4/6: tie -> higher
        self.assertEqual(choose_threshold(p, 0.0), 0.5)     # closest possible is the top value

    def test_threshold_labels_only_move_maybe(self) -> None:
        stated = np.array(["yes", "maybe", "no", "maybe"])
        probs = np.array([[0.6, 0.1, 0.3], [0.2, 0.5, 0.3], [0.1, 0.8, 0.1], [0.4, 0.4, 0.2]])
        self.assertEqual(threshold_labels(stated, probs, 0.3).tolist(), ["maybe", "maybe", "no", "yes"])

    def test_conclusion_effect(self) -> None:
        self.assertAlmostEqual(conclusion_effect(np.array(["maybe", "yes"]), np.array(["yes", "yes"])), 0.5)


class AnalyzeTests(unittest.TestCase):
    def test_runs_end_to_end_and_excludes_pilot(self) -> None:
        rng = np.random.default_rng(1)
        pmids = [str(i) for i in range(40)] + ["10135926"]   # one pilot question
        labels = ("yes", "no", "maybe")
        data = {p: {"final_decision": labels[i % 3], "reasoning_free_pred": labels[i % 3],
                    "reasoning_required_pred": labels[(i + 1) % 3]} for i, p in enumerate(pmids)}
        runs = {}
        for m in MODELS:
            for prompt in PROMPTS:
                for seat in ("with", "without"):
                    rows = {}
                    for p in pmids:
                        pm = float(rng.choice([0.0, 0.1, 0.3, 0.6]))
                        rows[p] = {"label": "maybe" if pm >= 0.6 else "yes",
                                   "probs": {"yes": round(1 - pm, 4), "no": 0.0, "maybe": pm}}
                    runs[(m, prompt, seat)] = rows
        test = {str(i) for i in range(20)} | {"10135926"}
        result = analyze(runs, data, test, np.random.default_rng(0), n_boot=30)
        self.assertEqual((result["n_questions"], result["n_test"]), (40, 20))
        for key in ("C1_conclusion_effect_caveat_minus_probs", "T1_maybe_f1_vs_counterpart_threshold_minus_stated"):
            self.assertIn(result[key]["verdict"], ("supported", "refuted", "inconclusive"))
        self.assertEqual(len(result["thresholds"]), 8)


if __name__ == "__main__":
    unittest.main()
