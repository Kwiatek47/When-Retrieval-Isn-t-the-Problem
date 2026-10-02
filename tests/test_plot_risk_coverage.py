from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET

SIGNAL = {
    "n": 10,
    "errors": 3,
    "always_answer_cost": 0.3,
    "aurc": {"value": 0.2},
    "aurc_oracle": 0.05,
    "operating_point": {"abstention_rate": 0.3, "selective_accuracy": 0.857},
    # two errors late, one early: risk for k = 1..10
    "risk_by_k": [0.0, 0.5, 0.3333, 0.25, 0.2, 0.1667, 0.1429, 0.125, 0.2222, 0.3],
}


class RiskCoverageFigureTests(unittest.TestCase):
    def test_curve_ends_at_the_error_rate_and_skips_the_noisy_start(self) -> None:
        from scripts.agents.plot_risk_coverage import curve_points

        points = curve_points(SIGNAL["risk_by_k"], min_coverage=0.3)
        self.assertEqual(points[0], (0.3, 0.3333))
        self.assertEqual(points[-1], (1.0, 0.3))
        self.assertEqual(len(points), 8)

    def test_oracle_is_zero_until_correct_answers_run_out(self) -> None:
        from scripts.agents.plot_risk_coverage import oracle_points

        points = dict(oracle_points(10, 3, min_coverage=0.1))
        self.assertEqual(points[0.7], 0.0)
        self.assertAlmostEqual(points[0.8], 1 / 8)
        self.assertAlmostEqual(points[1.0], 0.3)

    def test_operating_point_is_coverage_and_risk(self) -> None:
        from scripts.agents.plot_risk_coverage import operating_point

        coverage, risk = operating_point(SIGNAL)
        self.assertAlmostEqual(coverage, 0.7)
        self.assertAlmostEqual(risk, 0.143)

    def test_svg_is_well_formed_and_states_the_numbers(self) -> None:
        from scripts.agents.plot_risk_coverage import render_svg

        svg = render_svg(SIGNAL, "BioLinkBERT 1 \u2212 confidence")
        root = ET.fromstring(svg)
        self.assertTrue(root.tag.endswith("svg"))
        text = " ".join(t.text or "" for t in root.iter() if t.tag.endswith("text"))
        self.assertIn("AURC 0.200", text)
        self.assertIn("random order: 0.300", text)
        self.assertIn("30.0% abstained", text)

    def test_pgfplots_has_balanced_braces_and_the_full_coverage_point(self) -> None:
        from scripts.agents.plot_risk_coverage import render_pgfplots

        tex = render_pgfplots(SIGNAL, "BioLinkBERT 1 \u2212 confidence", step=3)
        self.assertEqual(tex.count("{"), tex.count("}"))
        self.assertIn("(1.000,0.3000)", tex)
        self.assertIn("BioLinkBERT 1 $-$ confidence", tex)
        self.assertNotIn("\u2212", tex)


if __name__ == "__main__":
    unittest.main()
