from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


class TrainingDataTests(unittest.TestCase):
    def _write(self, tmp: str) -> tuple[Path, Path]:
        pqal = {
            "1": {"QUESTION": "Q1", "CONTEXTS": ["a", " ", "b"], "final_decision": "maybe"},
            "2": {"QUESTION": "Q2", "CONTEXTS": ["c"], "final_decision": "yes"},
            "3": {"QUESTION": "Q3", "CONTEXTS": ["d"], "final_decision": "no"},
        }
        test = [{"relevant_pmids": ["2"]}]
        pqal_path, test_path = Path(tmp) / "pqal.json", Path(tmp) / "test.json"
        pqal_path.write_text(json.dumps(pqal), encoding="utf-8")
        test_path.write_text(json.dumps(test), encoding="utf-8")
        return pqal_path, test_path

    def test_test_questions_never_enter_training(self) -> None:
        from scripts.classifier.train_maybe_detector import load_split

        with tempfile.TemporaryDirectory() as tmp:
            train, test = load_split(*self._write(tmp))
        self.assertEqual([e["pmid"] for e in train], ["1", "3"])
        self.assertEqual([e["pmid"] for e in test], ["2"])
        self.assertEqual(train[0]["evidence"], "a b")

    def test_label_ids_for_both_heads(self) -> None:
        from scripts.classifier.train_maybe_detector import label_ids

        examples = [{"label": "yes"}, {"label": "no"}, {"label": "maybe"}]
        self.assertEqual(label_ids(examples, "binary"), [0, 0, 1])
        self.assertEqual(label_ids(examples, "three_class"), [0, 1, 2])

    def test_balanced_sampling_gives_each_class_equal_total_weight(self) -> None:
        from scripts.classifier.train_maybe_detector import sampling_weights

        ids = [0] * 8 + [1] * 2
        self.assertIsNone(sampling_weights(ids, "natural"))
        weights = sampling_weights(ids, "balanced")
        self.assertAlmostEqual(sum(weights[:8]), sum(weights[8:]))
        with self.assertRaises(ValueError):
            sampling_weights(ids, "oversample")


class MetricTests(unittest.TestCase):
    def test_average_precision_matches_the_stepwise_definition(self) -> None:
        from scripts.agents.analyze_rq8_maybe_detector import average_precision

        scores = np.array([0.9, 0.8, 0.7, 0.6])
        positive = np.array([True, False, True, False])
        # recall steps 0.5 at precision 1/1 and 0.5 at precision 2/3
        self.assertAlmostEqual(average_precision(scores, positive), 0.5 * 1.0 + 0.5 * (2 / 3))
        self.assertAlmostEqual(average_precision(scores, np.array([True, True, False, False])), 1.0)

    def test_average_precision_of_a_constant_score_is_the_base_rate(self) -> None:
        from scripts.agents.analyze_rq8_maybe_detector import average_precision

        positive = np.array([True] * 11 + [False] * 89)
        self.assertAlmostEqual(average_precision(np.zeros(100), positive), 0.11)

    def test_precision_at_hits(self) -> None:
        from scripts.agents.analyze_rq8_maybe_detector import precision_at_hits

        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        positive = np.array([True, False, True, False, True])
        self.assertAlmostEqual(precision_at_hits(scores, positive, hits=2), 2 / 3)
        self.assertTrue(np.isnan(precision_at_hits(scores, positive, hits=4)))

    def test_effects_are_main_effects_of_the_two_by_two(self) -> None:
        from scripts.agents.analyze_rq8_maybe_detector import average_precision, balance_effect, head_effect

        rng = np.random.default_rng(0)
        positive = rng.random(60) < 0.3
        bn, bb, tn, tb = (rng.random(60) for _ in range(4))
        ap = lambda s: average_precision(s, positive)  # noqa: E731
        self.assertAlmostEqual(head_effect(bn, bb, tn, tb, positive), (ap(bn) + ap(bb) - ap(tn) - ap(tb)) / 2)
        self.assertAlmostEqual(balance_effect(bn, bb, tn, tb, positive), (ap(bb) + ap(tb) - ap(bn) - ap(tn)) / 2)

    def test_cells_are_averaged_over_seeds_and_aligned_by_pmid(self) -> None:
        from scripts.agents.analyze_rq8_maybe_detector import HEADS, SAMPLINGS, load_cells

        with tempfile.TemporaryDirectory() as tmp:
            for head in HEADS:
                for sampling in SAMPLINGS:
                    for seed, value in ((1, 0.2), (2, 0.4)):
                        rows = [{"meta": {}}, {"pmid": "b", "p_maybe": value}, {"pmid": "a", "p_maybe": value / 2}]
                        (Path(tmp) / f"{head}.{sampling}.seed{seed}.jsonl").write_text(
                            "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
                        )
            pmids, ensemble, seed_arrays = load_cells(Path(tmp), seeds=(1, 2))
        self.assertEqual(pmids, ["a", "b"])
        np.testing.assert_allclose(ensemble[("binary", "natural")], [0.15, 0.3])
        self.assertEqual(len(seed_arrays[("three_class", "balanced")]), 2)


if __name__ == "__main__":
    unittest.main()
