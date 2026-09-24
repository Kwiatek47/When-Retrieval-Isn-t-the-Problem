from __future__ import annotations

import itertools
import unittest

import numpy as np


class HedgeLexiconTests(unittest.TestCase):
    def test_counts_epistemic_hedges_and_calls_for_more_research(self) -> None:
        from scripts.agents.build_pqal_label_table import count_cues

        text = (
            "HER2 might have a limited prognostic value. These data suggest a role, "
            "but further prospective studies are needed and the mechanism remains to be established."
        )
        # might, suggest, further ... studies, remains to be
        self.assertEqual(count_cues(text, "hedge"), 4)
        # the extended list adds "limited"
        self.assertEqual(count_cues(text, "hedge_ext"), 5)

    def test_month_may_is_not_a_hedge(self) -> None:
        from scripts.agents.build_pqal_label_table import count_cues

        self.assertEqual(count_cues("Patients were enrolled from May 2010 to June 2012.", "hedge"), 0)
        self.assertEqual(count_cues("Screening may reduce mortality.", "hedge"), 1)
        self.assertEqual(count_cues("Results were mixed. May this apply to children?", "hedge"), 1)

    def test_null_results_are_counted_separately_from_hedges(self) -> None:
        from scripts.agents.build_pqal_label_table import count_cues

        text = "The difference was not statistically significant and the drug failed to reduce pain."
        self.assertEqual(count_cues(text, "null"), 2)
        self.assertEqual(count_cues(text, "hedge"), 0)

    def test_results_text_keeps_only_results_sections(self) -> None:
        from scripts.agents.build_pqal_label_table import results_text

        contexts = ["We asked whether X may help.", "120 patients.", "X reduced Y (p=0.01)."]
        labels = ["OBJECTIVE", "METHODS", "RESULTS"]
        self.assertEqual(results_text(contexts, labels), "X reduced Y (p=0.01).")
        self.assertEqual(results_text(contexts[:2], labels[:2]), "")

    def test_build_row_records_annotators_and_who_the_final_label_follows(self) -> None:
        from scripts.agents.build_pqal_label_table import build_row

        item = {
            "QUESTION": "Does HER2 predict outcome?",
            "CONTEXTS": ["Aim.", "HER2 was significant in univariate analysis."],
            "LABELS": ["OBJECTIVE", "RESULTS"],
            "LONG_ANSWER": "HER2 might have a limited prognostic value.",
            "reasoning_free_pred": "maybe",
            "reasoning_required_pred": "yes",
            "final_decision": "maybe",
            "YEAR": "2007",
        }
        row = build_row("17940352", item, "cv")
        self.assertEqual(row["pattern"], "rf=maybe|rr=yes")
        self.assertEqual(row["final_follows"], "rf")
        self.assertFalse(row["all_agree"])
        self.assertEqual(row["hedge_count_conclusion"], 1)
        self.assertEqual(row["hedge_count_results"], 0)
        self.assertTrue(row["has_results_section"])


class AurocTests(unittest.TestCase):
    def _brute_force(self, scores: list[float], positive: list[bool]) -> float:
        pos = [s for s, p in zip(scores, positive) if p]
        neg = [s for s, p in zip(scores, positive) if not p]
        wins = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a, b in itertools.product(pos, neg))
        return wins / (len(pos) * len(neg))

    def test_rank_auroc_matches_pairwise_definition_with_ties(self) -> None:
        from scripts.agents.analyze_h1_hedging import auroc

        rng = np.random.default_rng(0)
        for _ in range(20):
            scores = rng.integers(0, 4, size=40).astype(float)  # many ties, like hedge counts
            positive = rng.random(40) < 0.3
            if positive.all() or not positive.any():
                continue
            self.assertAlmostEqual(
                auroc(scores, positive), self._brute_force(list(scores), list(positive)), places=10
            )

    def test_auroc_edge_cases(self) -> None:
        from scripts.agents.analyze_h1_hedging import auroc

        self.assertEqual(auroc(np.array([0.0, 1.0]), np.array([False, True])), 1.0)
        self.assertEqual(auroc(np.array([1.0, 1.0]), np.array([False, True])), 0.5)
        self.assertTrue(np.isnan(auroc(np.array([1.0, 2.0]), np.array([True, True]))))


if __name__ == "__main__":
    unittest.main()
