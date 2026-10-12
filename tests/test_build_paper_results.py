from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.agents.build_paper_results import check_all, check_claim, copy_all, same


class SameTests(unittest.TestCase):
    def test_rounding_to_the_claims_decimals(self) -> None:
        self.assertTrue(same(0.2754, "0.275"))
        self.assertTrue(same(-0.1558, "-0.156"))
        self.assertTrue(same(299, "299"))
        self.assertFalse(same(0.2764, "0.275"))
        self.assertFalse(same(298, "299"))

    def test_a_stored_tie_accepts_either_neighbour(self) -> None:
        self.assertTrue(same(0.1725, "0.172"))
        self.assertTrue(same(0.1725, "0.173"))


class CheckClaimTests(unittest.TestCase):
    doc = {"gap": {"value": 0.2754, "ci_low": 0.1419, "ci_high": 0.427, "verdict": "supported"},
           "cells": {"a": 3, "b": 4}}

    def test_value_interval_and_verdict(self) -> None:
        claim = {"id": "X", "path": "/gap", "value": "0.275", "ci": ["0.142", "0.427"], "verdict": "supported"}
        self.assertEqual(check_claim(claim, self.doc), [])

    def test_each_mismatch_is_reported(self) -> None:
        claim = {"id": "X", "path": "/gap", "value": "0.300", "ci": ["0.142", "0.500"], "verdict": "refuted"}
        problems = check_claim(claim, self.doc)
        self.assertEqual(len(problems), 3)

    def test_sum_of_paths(self) -> None:
        self.assertEqual(check_claim({"id": "X", "sum": ["/cells/a", "/cells/b"], "value": "7"}, self.doc), [])


class CheckAllTests(unittest.TestCase):
    def test_stale_copy_and_missing_key_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "reports").mkdir()
            (root / "reports/r.json").write_text(json.dumps({"n": 5}), encoding="utf-8")
            (root / "reports/f.svg").write_text("<svg/>", encoding="utf-8")
            spec = {
                "sources": {"r": {"from": "reports/r.json"}},
                "figures": {"f.svg": "reports/f.svg"},
                "sections": [{"claims": [{"id": "A", "source": "r", "path": "/n", "value": "5"},
                                         {"id": "B", "source": "r", "path": "/missing", "value": "1"}]}],
            }
            results = root / "paper/results"
            copy_all(spec, root=root, results=results)
            failures, stale = check_all(spec, root=root, results=results)
            self.assertEqual([f.split(":")[0] for f in failures], ["B"])
            self.assertEqual(stale, [])

            (root / "reports/r.json").write_text(json.dumps({"n": 6}), encoding="utf-8")
            _, stale = check_all(spec, root=root, results=results)
            self.assertEqual(stale, ["reports/r.json"])


if __name__ == "__main__":
    unittest.main()
