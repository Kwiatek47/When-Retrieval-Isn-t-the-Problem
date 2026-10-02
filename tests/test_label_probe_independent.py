from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


def _arr(text: str) -> np.ndarray:
    """Labels from a compact string: y = yes, n = no, m = maybe."""
    return np.array([{"y": "yes", "n": "no", "m": "maybe"}[c] for c in text])


class IndependentReferenceTests(unittest.TestCase):
    GOLD = _arr("mmmmyynn")
    RR = _arr("mmmyyynn")     # vs gold: 3 hits, 3 predicted -> 6/7
    RF = _arr("mymmynnn")     # vs RR:   2 hits, 3 + 3      -> 4/6
    MODEL = _arr("myyymynn")  # vs gold: 1 hit, 2 predicted -> 2/6; vs RF: 1 hit, 2 + 3 -> 2/5

    def test_same_seat_gap_scores_both_readers_against_the_other_annotator(self) -> None:
        from scripts.agents.analyze_label_probe import gap_to_human

        self.assertAlmostEqual(gap_to_human(self.MODEL, self.RR, self.RF), 4 / 6 - 2 / 5)

    def test_label_gap_share_is_the_difference_of_the_two_gaps(self) -> None:
        from scripts.agents.analyze_label_probe_independent import label_gap_share

        against_final = 6 / 7 - 2 / 6
        against_independent = 4 / 6 - 2 / 5
        self.assertAlmostEqual(
            label_gap_share(self.MODEL, self.RR, self.GOLD, self.RF), against_final - against_independent
        )

    def test_agreement_gap(self) -> None:
        from scripts.agents.analyze_label_probe_independent import agreement_gap

        human = float((self.RR == self.RF).mean())
        model = float((self.MODEL == self.RF).mean())
        self.assertAlmostEqual(agreement_gap(self.MODEL, self.RR, self.RF), human - model)

    def test_pilot_questions_are_left_out_and_test_questions_never_enter(self) -> None:
        from scripts.agents.analyze_label_probe_independent import PILOT_N, PQAL, confirmatory_pmids
        from scripts.agents.run_label_probe import select_pmids

        if not PQAL.exists():
            self.skipTest("PQA-L not available")
        data = json.loads(PQAL.read_text(encoding="utf-8"))
        cv = select_pmids(data, "cv")
        pmids = confirmatory_pmids(data)
        self.assertEqual(pmids, cv[PILOT_N:])
        self.assertFalse(set(pmids) & set(select_pmids(data, "test")))

    def test_unfinished_run_is_refused(self) -> None:
        from scripts.agents.analyze_label_probe_independent import require_complete

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            path.write_text(json.dumps({"pmid": "1", "repeat": 0, "label": None}) + '\n{"partial', encoding="utf-8")
            require_complete(path, ["1"])  # a recorded failure is a finished question
            with self.assertRaises(SystemExit):
                require_complete(path, ["1", "2"])

    def test_analyze_mode_runs_end_to_end_on_synthetic_data(self) -> None:
        from scripts.agents.analyze_label_probe_independent import analyze_mode

        rng = np.random.default_rng(0)
        labels = np.array(["yes", "no", "maybe"])
        frame = {k: labels[rng.integers(0, 3, size=80)] for k in ("gold", "rr", "rf")}
        without, with_conclusion = labels[rng.integers(0, 3, size=80)], labels[rng.integers(0, 3, size=80)]
        r = analyze_mode(without, with_conclusion, frame, np.random.default_rng(1), 40)
        self.assertIn(r["S1_same_seat_gap"]["verdict"], ("model below human", "model above human", "not distinguishable"))
        self.assertIn(r["S2_gap_owed_to_co_created_label"]["verdict"], ("supported", "refuted", "inconclusive"))
        self.assertAlmostEqual(
            r["S2_gap_owed_to_co_created_label"]["value"],
            round(r["R1_gap_against_final_label"]["value"] - r["S1_same_seat_gap"]["value"], 4),
            places=3,
        )


if __name__ == "__main__":
    unittest.main()
