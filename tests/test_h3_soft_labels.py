from __future__ import annotations

import unittest

import numpy as np


def _arr(text: str) -> np.ndarray:
    """Labels from a compact string: y = yes, n = no, m = maybe."""
    return np.array([{"y": "yes", "n": "no", "m": "maybe"}[c] for c in text])


class SoftLabelTests(unittest.TestCase):
    RF = _arr("yymn")
    RR = _arr("ynmm")

    def test_soft_label_is_the_mean_of_the_annotations(self) -> None:
        from scripts.agents.analyze_h3_soft_labels import soft_labels

        q = soft_labels(self.RF, self.RR)
        # columns: yes, no, maybe
        np.testing.assert_allclose(q, [[1, 0, 0], [0.5, 0.5, 0], [0, 0, 1], [0, 0.5, 0.5]])
        np.testing.assert_allclose(q.sum(axis=1), 1.0)
        three = soft_labels(self.RF, self.RR, _arr("yymm"))
        np.testing.assert_allclose(three[1], [2 / 3, 1 / 3, 0])

    def test_soft_accuracy_gives_partial_credit_on_split_questions(self) -> None:
        from scripts.agents.analyze_h3_soft_labels import soft_accuracy, soft_labels

        q = soft_labels(self.RF, self.RR)
        # yes (1.0), yes (0.5), maybe (1.0), yes (0.0)
        self.assertAlmostEqual(soft_accuracy(_arr("yymy"), q), 2.5 / 4)
        # where annotators agree, soft accuracy equals hard accuracy
        agreed = soft_labels(_arr("ynm"), _arr("ynm"))
        self.assertAlmostEqual(soft_accuracy(_arr("ynn"), agreed), 2 / 3)

    def test_interaction_is_zero_when_labels_are_not_soft(self) -> None:
        from scripts.agents.analyze_h3_soft_labels import interaction, one_hot

        gold = _arr("yynm")
        self.assertAlmostEqual(interaction(_arr("yynn"), _arr("ynnm"), gold, one_hot(gold)), 0.0)

    def test_interaction_and_order_flip(self) -> None:
        from scripts.agents.analyze_h3_soft_labels import interaction, order_differs, soft_labels

        q = soft_labels(self.RF, self.RR)
        gold = _arr("yymm")
        a, b = _arr("ynmm"), _arr("yymn")
        # hard: a 3/4, b 3/4 -> 0; soft: a = 1+.5+1+.5 = 3, b = 1+.5+1+.5 = 3 -> 0
        self.assertAlmostEqual(interaction(a, b, gold, q), 0.0)
        c = _arr("ynmn")  # hard 2/4; soft 1 + .5 + 1 + .5 = 3/4
        self.assertAlmostEqual(interaction(c, b, gold, q), 0.0 - (-0.25))
        self.assertEqual(order_differs(c, b, gold, q), 1.0)  # behind on hard, level on soft
        self.assertEqual(order_differs(a, b, gold, q), 0.0)

    def test_soft_maybe_counts_probability_mass(self) -> None:
        from scripts.agents.analyze_h3_soft_labels import soft_labels, soft_maybe_prf

        q = soft_labels(self.RF, self.RR)  # maybe mass: 0, 0, 1, 0.5 -> 1.5
        prf = soft_maybe_prf(_arr("yymm"), q)
        self.assertEqual((prf["soft_true_positives"], prf["predicted"], prf["gold_mass"]), (1.5, 2, 1.5))
        self.assertEqual((prf["precision"], prf["recall"]), (0.75, 1.0))
        self.assertAlmostEqual(prf["f1"], round(3 / 3.5, 4))

    def test_verdict_follows_the_registered_rule(self) -> None:
        from scripts.agents.analyze_h3_soft_labels import verdict

        self.assertEqual(verdict({"ci_low": 0.005, "ci_high": 0.03}), "supported")
        self.assertEqual(verdict({"ci_low": -0.04, "ci_high": -0.001}), "supported")
        self.assertEqual(verdict({"ci_low": -0.01, "ci_high": 0.015}), "refuted")
        self.assertEqual(verdict({"ci_low": -0.01, "ci_high": 0.05}), "inconclusive")

    def test_analyze_runs_end_to_end_on_a_synthetic_frame(self) -> None:
        from scripts.agents.analyze_h2_human_ceiling import SYSTEMS
        from scripts.agents.analyze_h3_soft_labels import PRIMARY_PAIR, analyze, soft_labels

        rng = np.random.default_rng(0)
        labels = np.array(["yes", "no", "maybe"])
        frame = {key: labels[rng.integers(0, 3, size=60)] for key in ("gold", "rf", "rr", *SYSTEMS)}
        result = analyze(frame, soft_labels(frame["rf"], frame["rr"]), np.random.default_rng(1), 50)
        self.assertIn(" - ".join(PRIMARY_PAIR), result["pairs"])
        self.assertEqual(len(result["pairs"]), 6)
        self.assertIn("always_yes", result["systems"])
        primary = result["pairs"][" - ".join(PRIMARY_PAIR)]
        self.assertAlmostEqual(
            primary["interaction"]["value"], round(primary["soft_difference"] - primary["hard_difference"], 4), places=3
        )


if __name__ == "__main__":
    unittest.main()
