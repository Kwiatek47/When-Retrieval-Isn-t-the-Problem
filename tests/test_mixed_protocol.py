from __future__ import annotations

import unittest

import numpy as np

from scripts.agents.analyze_mixed_protocol import analyze, dispute_rate, mixed_minus_qq


def _row(pmid: str, with_label: str, without_label: str, final: str | None, status: str) -> dict:
    follows = "both" if with_label == without_label == final else (
        "with_conclusion" if final == with_label else "context_only" if final == without_label else "none")
    return {"pmid": pmid, "status": status, "final": final, "final_follows": follows,
            "initial": {"with_conclusion": with_label, "context_only": without_label}}


class StatisticTests(unittest.TestCase):
    def test_dispute_rate(self) -> None:
        self.assertAlmostEqual(dispute_rate(np.array(["yes", "no", "maybe"]), np.array(["yes", "yes", "maybe"])), 1 / 3)

    def test_mixed_minus_qq_averages_the_two_orders(self) -> None:
        qq = (np.array(["yes", "yes"]), np.array(["yes", "yes"]))      # 0 disputes
        qm = (np.array(["yes", "no"]), np.array(["yes", "yes"]))       # 1/2
        mq = (np.array(["no", "no"]), np.array(["yes", "yes"]))        # 2/2
        self.assertAlmostEqual(mixed_minus_qq(dispute_rate)(*qq, *qm, *mq), 0.75)


class AnalyzeTests(unittest.TestCase):
    def test_paired_on_questions_every_run_completed(self) -> None:
        data = {str(i): {"final_decision": g, "reasoning_free_pred": a, "reasoning_required_pred": b}
                for i, (a, b, g) in enumerate([("maybe", "yes", "maybe"), ("yes", "yes", "yes"),
                                               ("no", "no", "no"), ("maybe", "maybe", "maybe")])}
        qq = [_row("0", "yes", "yes", "yes", "agreed"), _row("1", "yes", "yes", "yes", "agreed"),
              _row("2", "no", "no", "no", "agreed"), _row("3", "yes", "yes", "yes", "agreed")]
        qm = [_row("0", "yes", "maybe", "maybe", "negotiated"), _row("1", "yes", "yes", "yes", "agreed"),
              _row("2", "no", "no", "no", "agreed"), _row("3", "maybe", "maybe", "maybe", "agreed")]
        mq = [_row("0", "maybe", "yes", "maybe", "negotiated"), _row("1", "yes", "yes", "yes", "agreed"),
              _row("2", "no", "no", "no", "agreed"), {"pmid": "3", "status": "failed"}]
        result = analyze({"QQ": qq, "QM": qm, "MQ": mq}, data, np.random.default_rng(0), n_boot=50)
        self.assertEqual(result["n_questions"], 3)  # question 3 failed in MQ
        self.assertAlmostEqual(result["M1_dispute_rate_mixed_minus_QQ"]["value"], round(1 / 3, 4))
        self.assertEqual(result["round0_maybe_by_model_and_seat"]["mistral_with_conclusion"], 1)
        self.assertEqual(result["runs"]["QQ"]["final_label_counts"], {"yes": 2, "no": 1})


if __name__ == "__main__":
    unittest.main()
