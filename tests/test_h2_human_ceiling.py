from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


def _arr(text: str) -> np.ndarray:
    """Labels from a compact string: y = yes, n = no, m = maybe."""
    return np.array([{"y": "yes", "n": "no", "m": "maybe"}[c] for c in text])


class H2StatisticTests(unittest.TestCase):
    # Six questions. RR departs from gold on the last three (disputed).
    GOLD = _arr("yynmmm")
    RR = _arr("yynyny")
    #            agreed: all right | disputed: RR's label, RR's label, right
    PRED = _arr("yynynm")

    def test_error_gap_is_disputed_minus_agreed_error_rate(self) -> None:
        from scripts.agents.analyze_h2_human_ceiling import error_gap

        # agreed: 0/3 wrong; disputed: 2/3 wrong
        self.assertAlmostEqual(error_gap(self.PRED, self.GOLD, self.RR), 2 / 3)
        # a system that errs only where RR agreed with gold has a negative gap
        self.assertAlmostEqual(error_gap(_arr("nnymmm"), self.GOLD, self.RR), -1.0)

    def test_siding_counts_only_errors_on_disputed_questions(self) -> None:
        from scripts.agents.analyze_h2_human_ceiling import siding_with_rr

        # errors on disputed: "y" (= RR) and "n" (= RR) -> both side with RR
        self.assertAlmostEqual(siding_with_rr(self.PRED, self.GOLD, self.RR), 1.0)
        # "n" where RR said "y" is the third label
        self.assertAlmostEqual(siding_with_rr(_arr("yynnnm"), self.GOLD, self.RR), 0.5)
        self.assertTrue(np.isnan(siding_with_rr(self.GOLD, self.GOLD, self.RR)))

    def test_accuracy_shift_and_error_share(self) -> None:
        from scripts.agents.analyze_h2_human_ceiling import accuracy_shift, error_share_on_disputed

        # vs RR: 5/6 right; vs gold: 4/6 right
        self.assertAlmostEqual(accuracy_shift(self.PRED, self.GOLD, self.RR), 1 / 6)
        self.assertAlmostEqual(error_share_on_disputed(self.PRED, self.GOLD, self.RR), 1.0)

    def test_maybe_metrics(self) -> None:
        from scripts.agents.analyze_h2_human_ceiling import maybe_prf, maybe_recall_by_rr

        prf = maybe_prf(_arr("ymmn"), _arr("mmyn"))
        self.assertEqual((prf["hits"], prf["predicted"], prf["gold"]), (1, 2, 2))
        self.assertEqual((prf["precision"], prf["recall"], prf["f1"]), (0.5, 0.5, 0.5))

        split = maybe_recall_by_rr(_arr("mymy"), _arr("mmmm"), _arr("mmyy"))
        self.assertEqual(split, {"rr_maybe": {"n": 2, "hits": 1}, "rr_not_maybe": {"n": 2, "hits": 1}})

    def test_verdict_follows_the_registered_rule(self) -> None:
        from scripts.agents.analyze_h2_human_ceiling import verdict

        self.assertEqual(verdict({"ci_low": 0.01, "ci_high": 0.2}), "supported")
        self.assertEqual(verdict({"ci_low": -0.2, "ci_high": 0.0}), "refuted")
        self.assertEqual(verdict({"ci_low": -0.05, "ci_high": 0.1}), "inconclusive")

    def test_bootstrap_is_deterministic_and_brackets_the_point(self) -> None:
        from scripts.agents.analyze_h2_human_ceiling import bootstrap_ci, error_gap

        arrays = (np.tile(self.PRED, 30), np.tile(self.GOLD, 30), np.tile(self.RR, 30))
        a = bootstrap_ci(error_gap, arrays, np.random.default_rng(1), 200)
        b = bootstrap_ci(error_gap, arrays, np.random.default_rng(1), 200)
        self.assertEqual(a, b)
        self.assertLessEqual(a["ci_low"], a["value"])
        self.assertGreaterEqual(a["ci_high"], a["value"])


class H2LoadingTests(unittest.TestCase):
    def test_predictions_are_keyed_by_pmid(self) -> None:
        from scripts.agents.analyze_h2_human_ceiling import load_predictions

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            path.write_text(
                json.dumps({"cases": [{"id": "pubmedqa-official-123", "predicted_label": "maybe"}]}),
                encoding="utf-8",
            )
            self.assertEqual(load_predictions(path, "predicted_label"), {"123": "maybe"})


if __name__ == "__main__":
    unittest.main()
