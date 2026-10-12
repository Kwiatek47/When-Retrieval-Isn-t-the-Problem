from __future__ import annotations

import unittest


def _item(gold: str, context_only: str, sees_conclusion: str) -> dict:
    return {
        "final_decision": gold,
        "reasoning_required_pred": context_only,
        "reasoning_free_pred": sees_conclusion,
    }


class SampleTests(unittest.TestCase):
    def test_sample_is_gold_maybe_balanced_across_strata_and_reproducible(self) -> None:
        from scripts.agents.rq5_maybe_taxonomy import sample, stratum

        data = {f"u{i}": _item("maybe", "maybe", "maybe") for i in range(5)}
        data |= {f"n{i}": _item("maybe", "yes", "maybe") for i in range(5)}
        data |= {f"x{i}": _item("yes", "maybe", "maybe") for i in range(5)}

        chosen = sample(data, per_stratum=3, seed=47)

        self.assertEqual(chosen, sample(data, per_stratum=3, seed=47))
        self.assertEqual(len(set(chosen)), 6)
        self.assertTrue(all(data[p]["final_decision"] == "maybe" for p in chosen))
        self.assertEqual(sorted(stratum(data[p]) for p in chosen), ["negotiated"] * 3 + ["unanimous"] * 3)



class AnalysisTests(unittest.TestCase):
    def test_cohen_kappa_known_values(self) -> None:
        from scripts.agents.rq5_maybe_taxonomy import cohen_kappa

        self.assertAlmostEqual(cohen_kappa(list("AABB"), list("AABB")), 1.0)
        # observed 0.5, expected 0.5 -> 0
        self.assertAlmostEqual(cohen_kappa(list("AABB"), list("ABAB")), 0.0)
        # observed 0.8; expected (3*2 + 2*3)/25 = 0.48 -> (0.8 - 0.48) / 0.52
        self.assertAlmostEqual(cohen_kappa(list("AAABB"), list("AABBB")), (0.8 - 0.48) / 0.52)

    def test_disagreements_count_half_for_each_code(self) -> None:
        from scripts.agents.rq5_maybe_taxonomy import CLASSES, soft_share

        pairs = [("A", "A"), ("A", "F"), ("F", "F"), ("D", "E")]
        self.assertAlmostEqual(soft_share(pairs, CLASSES["content"]), (1 + 0.5 + 0 + 0) / 4)
        self.assertAlmostEqual(soft_share(pairs, CLASSES["task"]), 1 / 4)

    def test_permutation_detects_a_clear_difference_and_not_a_null(self) -> None:
        from scripts.agents.rq5_maybe_taxonomy import CLASSES, permutation_p

        content = CLASSES["content"]
        clear = permutation_p([("A", "A")] * 20, [("F", "F")] * 20, content, 2000, 1)
        null = permutation_p([("A", "A"), ("F", "F")] * 10, [("F", "F"), ("A", "A")] * 10, content, 2000, 1)
        self.assertLess(clear, 0.01)
        self.assertGreater(null, 0.5)

    def test_verdict_respects_the_kappa_gate(self) -> None:
        from scripts.agents.rq5_maybe_taxonomy import primary_verdict

        strong = {"permutation_p": 0.001, "difference": 0.4}
        self.assertEqual(primary_verdict(0.39, strong), "not interpretable (kappa below gate)")
        self.assertEqual(primary_verdict(float("nan"), strong), "not interpretable (kappa below gate)")
        self.assertEqual(primary_verdict(0.6, strong), "supported")
        self.assertEqual(primary_verdict(0.6, {"permutation_p": 0.001, "difference": -0.4}), "contradicted")
        self.assertEqual(primary_verdict(0.6, {"permutation_p": 0.2, "difference": 0.4}), "inconclusive")

    def test_population_distribution_reweights_strata(self) -> None:
        from scripts.agents.rq5_maybe_taxonomy import population_distribution

        dist = population_distribution({"unanimous": [("A", "A")] * 20, "negotiated": [("F", "F")] * 20})
        self.assertAlmostEqual(dist["content"], round(23 / 110, 3))
        self.assertAlmostEqual(dist["annotator"], round(87 / 110, 3))

    def test_analyze_end_to_end_on_synthetic_files(self) -> None:
        import csv
        import json
        from pathlib import Path
        import tempfile

        from scripts.agents import rq5_maybe_taxonomy as rq5

        with tempfile.TemporaryDirectory() as tmp:
            key = [{"item": i, "stratum": "unanimous" if i <= 4 else "negotiated"} for i in range(1, 9)]
            key_path = Path(tmp) / "key.json"
            key_path.write_text(json.dumps(key), encoding="utf-8")
            files = []
            for name, codes in (("c1", "AABBFFFD"), ("c2", "AACBFFED"), ("llm", "AABBFFFF")):
                path = Path(tmp) / f"{name}.csv"
                with path.open("w", encoding="utf-8", newline="") as fh:
                    w = csv.writer(fh)
                    w.writerow(["item", "code"])
                    w.writerows([[i, c] for i, c in enumerate(codes, start=1)])
                files.append(path)
            original = rq5.KEY
            rq5.KEY = key_path
            try:
                result = rq5.analyze(files[0], files[1], files[2], n_perm=500, seed=1)
            finally:
                rq5.KEY = original
        self.assertEqual(result["agreement_six_codes"], 0.75)
        self.assertTrue(result["gate"]["passed"])
        self.assertEqual(result["primary_content_uncertainty"]["unanimous_share"], 1.0)
        self.assertIn(result["primary_content_uncertainty"]["verdict"], ("supported", "inconclusive"))
        self.assertIn("kappa_with_coder_1", result["llm_coder"])


if __name__ == "__main__":
    unittest.main()
