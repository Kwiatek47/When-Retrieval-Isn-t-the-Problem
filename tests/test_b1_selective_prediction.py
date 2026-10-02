from __future__ import annotations

import unittest

import numpy as np


class RiskCoverageTests(unittest.TestCase):
    # five questions, least to most uncertain; errors at positions 2 and 4
    UNC = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    WRONG = np.array([False, True, False, True, False])

    def test_risk_after_answering_the_k_least_uncertain(self) -> None:
        from scripts.agents.analyze_b1_selective_prediction import aurc, risk_by_coverage

        risks = risk_by_coverage(self.UNC, self.WRONG)
        np.testing.assert_allclose(risks, [0, 1 / 2, 1 / 3, 2 / 4, 2 / 5])
        self.assertAlmostEqual(risks[-1], self.WRONG.mean())  # full coverage = the error rate
        self.assertAlmostEqual(aurc(self.UNC, self.WRONG), np.mean([0, 1 / 2, 1 / 3, 2 / 4, 2 / 5]))

    def test_order_of_input_does_not_matter(self) -> None:
        from scripts.agents.analyze_b1_selective_prediction import aurc

        shuffle = np.array([3, 0, 4, 1, 2])
        self.assertAlmostEqual(aurc(self.UNC[shuffle], self.WRONG[shuffle]), aurc(self.UNC, self.WRONG))

    def test_ties_share_errors_so_a_constant_signal_scores_the_error_rate(self) -> None:
        from scripts.agents.analyze_b1_selective_prediction import aurc, risk_by_coverage

        constant = np.zeros(5)
        np.testing.assert_allclose(risk_by_coverage(constant, self.WRONG), [0.4] * 5)
        self.assertAlmostEqual(aurc(constant, self.WRONG), 0.4)

    def test_oracle_ranks_every_error_last(self) -> None:
        from scripts.agents.analyze_b1_selective_prediction import aurc, oracle_aurc

        # correct, correct, correct, then two errors: risks 0, 0, 0, 1/4, 2/5
        self.assertAlmostEqual(oracle_aurc(self.WRONG), np.mean([0, 0, 0, 1 / 4, 2 / 5]))
        self.assertLess(oracle_aurc(self.WRONG), aurc(self.UNC, self.WRONG))


class CostTests(unittest.TestCase):
    UNC = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    WRONG = np.array([False, False, False, False, False, True, True, True])

    def test_cost_counts_wrong_answers_and_abstentions(self) -> None:
        from scripts.agents.analyze_b1_selective_prediction import cost, cost_gain

        none = np.zeros(8, dtype=bool)
        self.assertAlmostEqual(cost(self.WRONG, none, 0.25), 3 / 8)
        last_three = self.UNC > 0.5
        self.assertAlmostEqual(cost(self.WRONG, last_three, 0.25), 3 * 0.25 / 8)
        self.assertAlmostEqual(cost_gain(self.WRONG, last_three), 3 / 8 - 3 * 0.25 / 8)

    def test_best_threshold_abstains_exactly_on_the_errors(self) -> None:
        from scripts.agents.analyze_b1_selective_prediction import best_threshold

        self.assertEqual(best_threshold(self.UNC, self.WRONG, 0.25, 0.5), 0.5)

    def test_minimum_coverage_limits_abstention(self) -> None:
        from scripts.agents.analyze_b1_selective_prediction import best_threshold

        wrong = np.array([False, True, True, True, True, True, True, True])
        # cheapest unconstrained choice answers one question; the constraint forces at least four
        self.assertEqual(best_threshold(self.UNC, wrong, 0.25, 0.5), 0.4)

    def test_no_abstention_when_it_cannot_pay(self) -> None:
        from scripts.agents.analyze_b1_selective_prediction import best_threshold

        self.assertEqual(best_threshold(self.UNC, np.zeros(8, dtype=bool), 0.25, 0.5), float("inf"))

    def test_out_of_fold_decisions_are_deterministic_and_cover_every_question(self) -> None:
        from scripts.agents.analyze_b1_selective_prediction import out_of_fold_abstentions

        rng = np.random.default_rng(0)
        unc = rng.random(200)
        wrong = rng.random(200) < unc * 0.6
        a = out_of_fold_abstentions(unc, wrong, 0.25, np.random.default_rng(47))
        b = out_of_fold_abstentions(unc, wrong, 0.25, np.random.default_rng(47))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (200,))
        self.assertGreaterEqual(1 - a.mean(), 0.4)  # roughly honours the coverage floor out of fold


class SameSeatSystemsTests(unittest.TestCase):
    def test_analyze_system_runs_end_to_end_and_is_consistent(self) -> None:
        from scripts.agents.analyze_label_probe import gap_to_human
        from scripts.agents.analyze_same_seat_systems import analyze_system

        rng = np.random.default_rng(0)
        labels = np.array(["yes", "no", "maybe"])
        frame = {k: labels[rng.integers(0, 3, size=90)] for k in ("gold", "rr", "rf")}
        pred = labels[rng.integers(0, 3, size=90)]
        r = analyze_system(pred, frame, np.random.default_rng(1), 40)
        self.assertAlmostEqual(
            r["S1_same_seat_gap"]["value"], round(gap_to_human(pred, frame["rr"], frame["rf"]), 4)
        )
        self.assertAlmostEqual(
            r["S2_gap_owed_to_co_created_label"]["value"],
            round(r["gap_against_final_label"]["value"] - r["S1_same_seat_gap"]["value"], 4),
            places=3,
        )


if __name__ == "__main__":
    unittest.main()
