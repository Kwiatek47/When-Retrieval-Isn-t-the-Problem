import unittest

from app.agents.uncertainty import risk_coverage_curve


class RiskCoverageCurveTest(unittest.TestCase):
    def test_curve_reaches_full_coverage_even_when_scores_hit_one(self):
        scores = [0.2, 1.0, 1.0, 0.5]
        curve = risk_coverage_curve(scores, ["yes", "no", "yes", "no"], ["yes", "no", "no", "no"])
        self.assertEqual(max(p["coverage"] for p in curve["points"]), 1.0)
        full = [p for p in curve["points"] if p["coverage"] == 1.0][0]
        self.assertAlmostEqual(full["selective_accuracy"], 0.75)


if __name__ == "__main__":
    unittest.main()
