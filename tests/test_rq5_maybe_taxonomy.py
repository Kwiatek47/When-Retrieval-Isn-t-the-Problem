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


if __name__ == "__main__":
    unittest.main()
