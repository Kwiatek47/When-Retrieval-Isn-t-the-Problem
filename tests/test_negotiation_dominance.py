from __future__ import annotations

import unittest

import numpy as np

from scripts.agents.analyze_negotiation_dominance import (
    analyze,
    information_advantage,
    model_advantage,
    verdict_above,
    verdict_below,
    win_with,
)

W, C = "with_conclusion", "context_only"


class StatisticTests(unittest.TestCase):
    def test_wins_count_only_settled_disputes(self) -> None:
        self.assertAlmostEqual(win_with(np.array([W, W, C, "none", "neither"])), 2 / 3)

    def test_model_and_information_advantage(self) -> None:
        qm = np.array([W] * 87 + [C] * 13)   # qwen3 holds the conclusion and wins 0.87
        mq = np.array([W] * 34 + [C] * 66)   # mistral holds it; qwen3 (context side) wins 0.66
        self.assertAlmostEqual(model_advantage(qm, mq), (0.87 - 0.34) / 2)
        self.assertAlmostEqual(information_advantage(qm, mq), (0.87 + 0.34) / 2 - 0.5)
        # pure information advantage: the conclusion side wins 0.7 whoever holds it -> D = 0
        same = np.array([W] * 7 + [C] * 3)
        self.assertAlmostEqual(model_advantage(same, same), 0.0)
        self.assertAlmostEqual(information_advantage(same, same), 0.2)

    def test_verdicts(self) -> None:
        self.assertEqual(verdict_below({"ci_low": -0.3, "ci_high": -0.01}), "supported")
        self.assertEqual(verdict_below({"ci_low": 0.0, "ci_high": 0.2}), "refuted")
        self.assertEqual(verdict_above({"ci_low": -0.1, "ci_high": 0.2}), "inconclusive")


class AnalyzeTests(unittest.TestCase):
    def test_end_to_end_on_disputes_only(self) -> None:
        data = {str(i): {"final_decision": "yes", "reasoning_free_pred": "yes", "reasoning_required_pred": "no",
                         "LONG_ANSWER": "It helps.", "CONTEXTS": ["c"], "LABELS": ["RESULTS"]} for i in range(12)}
        data["11"].update(final_decision="maybe", reasoning_free_pred="maybe")  # the human reference needs a maybe

        def row(p, status, final, follows):
            return {"pmid": p, "status": status, "final": final, "final_follows": follows, "rounds_used": 1, "turns": []}

        source = {run: {str(i): row(str(i), "negotiated" if i < 8 else "agreed", "yes", W if i < 6 else C if i < 8 else "both")
                        for i in range(12)} for run in ("QM", "MQ")}
        arms = {(run, arm): {str(i): row(str(i), "negotiated", "yes", W if (arm == "evidence" or i < 4) else C) for i in range(8)}
                for run in ("QM", "MQ") for arm in ("old", "evidence")}
        result = analyze(source, arms, data, np.random.default_rng(0), n_boot=50)
        self.assertEqual(result["n_disputes"], {"QM": 8, "MQ": 8})
        # old: conclusion side wins 4/8 in both runs; evidence: 8/8 -> I rises by 0.5, D unchanged
        self.assertAlmostEqual(result["N2_information_advantage_evidence_minus_old"]["value"], 0.5)
        self.assertAlmostEqual(result["N1_model_advantage_evidence_minus_old"]["value"], 0.0)


if __name__ == "__main__":
    unittest.main()
