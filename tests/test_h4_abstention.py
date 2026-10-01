from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


def _arr(text: str) -> np.ndarray:
    """Labels from a compact string: y = yes, n = no, m = maybe."""
    return np.array([{"y": "yes", "n": "no", "m": "maybe"}[c] for c in text])


class H4StatisticTests(unittest.TestCase):
    #              yes/no questions      | maybe questions
    GOLD = _arr("yyynnn" + "mm")
    PRED = _arr("yynnny" + "yn")  # errors: index 2, 5 (yes/no) and 6, 7 (maybe)
    # the signal flags both yes/no errors, and is low on the maybe errors
    UNC = np.array([0.1, 0.2, 0.9, 0.1, 0.2, 0.8, 0.05, 0.15])

    def test_a_uses_only_gold_yes_no_questions(self) -> None:
        from scripts.agents.analyze_h4_abstention import auroc_error_on_yes_no

        # both errors outrank all four correct answers
        self.assertEqual(auroc_error_on_yes_no(self.UNC, self.PRED, self.GOLD), 1.0)

    def test_b_uses_only_the_systems_errors(self) -> None:
        from scripts.agents.analyze_h4_abstention import auroc_maybe_among_errors, separation

        # among the four errors, both maybe errors are the least uncertain
        self.assertEqual(auroc_maybe_among_errors(self.UNC, self.PRED, self.GOLD), 0.0)
        self.assertEqual(separation(self.UNC, self.PRED, self.GOLD), 1.0)
        # a signal that is high exactly on maybe errors singles them out
        flipped = self.UNC.copy()
        flipped[[6, 7]] = 0.99
        self.assertEqual(auroc_maybe_among_errors(flipped, self.PRED, self.GOLD), 1.0)

    def test_verdict_follows_the_registered_rule(self) -> None:
        from scripts.agents.analyze_h4_abstention import verdict

        self.assertEqual(verdict({"ci_low": 0.55}, {"ci_low": 0.02, "ci_high": 0.3}), "supported")
        # D above 0 is not enough if the signal does not flag ordinary errors
        self.assertEqual(verdict({"ci_low": 0.48}, {"ci_low": 0.02, "ci_high": 0.3}), "inconclusive")
        self.assertEqual(verdict({"ci_low": 0.55}, {"ci_low": -0.3, "ci_high": 0.0}), "refuted")
        self.assertEqual(verdict({"ci_low": 0.55}, {"ci_low": -0.1, "ci_high": 0.2}), "inconclusive")

    def test_abstention_table_splits_avoided_errors(self) -> None:
        from scripts.agents import analyze_h4_abstention as h4

        row = [r for r in h4.abstention_table(self.UNC, self.PRED, self.GOLD) if r["abstention_rate"] == 0.5][0]
        # 4 most uncertain: indices 2, 5 (errors on yes/no) and 1, 4 (correct)
        self.assertEqual(row["abstained"], 4)
        self.assertEqual(row["abstained_gold_maybe"], 0)
        self.assertEqual((row["errors_avoided_on_yes_no"], row["errors_avoided_on_maybe"]), (2, 0))
        self.assertEqual(row["correct_answers_lost"], 2)
        # kept: indices 0, 3 correct and 6, 7 wrong
        self.assertEqual(row["selective_accuracy"], 0.5)

    def test_load_signal_aligns_cases_with_labels(self) -> None:
        from scripts.agents.analyze_h4_abstention import load_signal

        data = {
            "2": {"final_decision": "maybe", "reasoning_required_pred": "yes"},
            "1": {"final_decision": "no", "reasoning_required_pred": "no"},
        }
        cases = [
            {"id": "pubmedqa-official-2", "predicted_label": "yes", "agreement": 0.5},
            {"id": "pubmedqa-official-1", "predicted_label": "no", "agreement": 1.0},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "selfconsistency_qwen3_8b_k4_pqal500.json").write_text(
                json.dumps({"cases": cases}), encoding="utf-8"
            )
            frame = load_signal("self_consistency_1_minus_agreement", data, Path(tmp))
        self.assertEqual(list(frame["pred"]), ["no", "yes"])
        self.assertEqual(list(frame["gold"]), ["no", "maybe"])
        self.assertEqual(list(frame["rr"]), ["no", "yes"])
        np.testing.assert_allclose(frame["unc"], [0.0, 0.5])

    def test_analyze_signal_runs_end_to_end_on_synthetic_data(self) -> None:
        from scripts.agents.analyze_h4_abstention import analyze_signal

        rng = np.random.default_rng(0)
        labels = np.array(["yes", "no", "maybe"])
        frame = {
            "pred": labels[rng.integers(0, 3, size=80)],
            "gold": labels[rng.integers(0, 3, size=80)],
            "rr": labels[rng.integers(0, 3, size=80)],
            "unc": rng.random(80),
        }
        result = analyze_signal(frame, np.random.default_rng(1), 40)
        self.assertIn(result["D_separation"]["verdict"], ("supported", "refuted", "inconclusive"))
        self.assertEqual(len(result["abstention"]), 5)
        self.assertAlmostEqual(
            result["D_separation"]["value"],
            round(result["A_error_among_gold_yes_no"]["value"] - result["B_maybe_among_errors"]["value"], 4),
            places=3,
        )


if __name__ == "__main__":
    unittest.main()
