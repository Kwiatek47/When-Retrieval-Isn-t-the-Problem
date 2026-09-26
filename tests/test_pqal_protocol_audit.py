"""Tests for the PQA-L protocol audit and the RQ8b re-scoring (synthetic inputs only)."""

from __future__ import annotations

import csv
import json
import random
import tempfile
import unittest
from pathlib import Path

from scripts.agents.audit_pqal_labels import (
    _label_metrics,
    _paired_accuracy_delta,
    ceilings_section,
    load_table,
    protocol_section,
    rq8b_section,
)


def _row(pmid: str, gold: str, context_only: str, sees_conclusion: str, split: str = "test") -> dict:
    """A label-table row reduced to the fields the audit reads."""
    if context_only == sees_conclusion:
        pattern = "agree"
    elif gold == sees_conclusion:
        pattern = "dispute_final_follows_conclusion_annotator"
    elif gold == context_only:
        pattern = "dispute_final_follows_context_only_annotator"
    else:
        pattern = "dispute_final_follows_neither"
    return {
        "pmid": pmid,
        "split": split,
        "gold": gold,
        "gold_is_maybe": gold == "maybe",
        "context_only_pred": context_only,
        "sees_conclusion_pred": sees_conclusion,
        "annotators_agree": context_only == sees_conclusion,
        "agreement_pattern": pattern,
    }


# 2 agreements, 3 disputes; of the disputes 2 are resolved towards the conclusion-reading
# annotator and both of those end as gold `maybe`.
FIXTURE = [
    _row("1", "yes", "yes", "yes"),
    _row("2", "no", "no", "no"),
    _row("3", "maybe", "yes", "maybe"),
    _row("4", "maybe", "no", "maybe"),
    _row("5", "yes", "yes", "no"),
]


class ProtocolSectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.section = protocol_section(FIXTURE, random.Random(47), n_boot=200)

    def test_dispute_counts(self) -> None:
        self.assertEqual(self.section["n_items"], 5)
        self.assertEqual(self.section["n_disputes"], 3)
        self.assertAlmostEqual(self.section["dispute_rate"], 0.6)

    def test_share_of_disputes_following_the_conclusion_annotator(self) -> None:
        share = self.section["share_of_disputes_following_conclusion_annotator"]
        self.assertAlmostEqual(share["mean"], 2 / 3, places=4)
        self.assertEqual(share["n"], 3)

    def test_gold_maybe_is_decomposed_by_which_annotator_said_maybe(self) -> None:
        self.assertEqual(self.section["n_gold_maybe"], 2)
        self.assertEqual(self.section["gold_maybe_source"], {"conclusion_annotator_only": 2})
        self.assertEqual(self.section["gold_maybe_unanimous"], 0)
        self.assertEqual(self.section["gold_maybe_disputed"], 2)

    def test_unanimous_maybe_is_counted_as_such(self) -> None:
        section = protocol_section([_row("9", "maybe", "maybe", "maybe")], random.Random(1), 50)
        self.assertEqual(section["gold_maybe_unanimous"], 1)
        self.assertEqual(section["gold_maybe_source"], {"both_annotators": 1})

    def test_maybe_with_neither_annotator_saying_maybe_is_flagged(self) -> None:
        section = protocol_section([_row("9", "maybe", "yes", "no")], random.Random(1), 50)
        self.assertEqual(section["gold_maybe_source"], {"neither_annotator": 1})


class CeilingsSectionTests(unittest.TestCase):
    def test_only_the_context_only_annotator_is_marked_a_model_ceiling(self) -> None:
        section = ceilings_section(FIXTURE, random.Random(47), n_boot=200)
        self.assertTrue(section["context_only_annotator"]["is_ceiling_for_models"])
        self.assertFalse(section["sees_conclusion_annotator"]["is_ceiling_for_models"])

    def test_accuracy_matches_a_hand_count(self) -> None:
        section = ceilings_section(FIXTURE, random.Random(47), n_boot=200)
        # context-only annotator is right on pmids 1, 2, 5 -> 3/5
        self.assertAlmostEqual(section["context_only_annotator"]["overall_accuracy"]["mean"], 0.6)
        # conclusion-reading annotator is right on 1, 2, 3, 4 -> 4/5
        self.assertAlmostEqual(section["sees_conclusion_annotator"]["overall_accuracy"]["mean"], 0.8)


class LabelMetricsTests(unittest.TestCase):
    def test_recall_precision_and_support_per_class(self) -> None:
        gold = ["yes", "yes", "no", "maybe"]
        pred = ["yes", "no", "no", "yes"]
        m = _label_metrics(gold, pred, random.Random(47), n_boot=100)
        self.assertAlmostEqual(m["overall_accuracy"]["mean"], 0.5)
        self.assertAlmostEqual(m["per_class"]["yes"]["recall"]["mean"], 0.5)
        self.assertAlmostEqual(m["per_class"]["yes"]["precision"], 0.5)  # 1 of 2 predicted yes
        self.assertEqual(m["per_class"]["maybe"]["support"], 1)
        self.assertAlmostEqual(m["per_class"]["maybe"]["recall"]["mean"], 0.0)

    def test_precision_is_none_when_the_class_is_never_predicted(self) -> None:
        m = _label_metrics(["maybe"], ["yes"], random.Random(47), n_boot=50)
        self.assertIsNone(m["per_class"]["maybe"]["precision"])
        self.assertEqual(m["per_class"]["maybe"]["n_predicted"], 0)

    def test_class_with_no_support_has_no_recall(self) -> None:
        m = _label_metrics(["yes", "yes"], ["yes", "yes"], random.Random(47), n_boot=50)
        self.assertIsNone(m["per_class"]["maybe"]["recall"])
        self.assertEqual(m["per_class"]["maybe"]["support"], 0)


class PairedAccuracyDeltaTests(unittest.TestCase):
    def test_sign_follows_the_better_reference_label_set(self) -> None:
        # 40 items, so the bootstrap has enough to exclude 0 — at n=4 a 0.5 gap legitimately
        # does not, which is the interval doing its job rather than a bug.
        pred = ["yes"] * 40
        agrees_more = ["yes"] * 30 + ["no"] * 10
        agrees_less = ["yes"] * 10 + ["no"] * 30
        out = _paired_accuracy_delta(pred, agrees_more, agrees_less, random.Random(47), n_boot=500)
        self.assertAlmostEqual(out["value"], 0.75 - 0.25, places=4)
        self.assertFalse(out["brackets_zero"])

    def test_small_sample_keeps_zero_inside_the_interval(self) -> None:
        out = _paired_accuracy_delta(
            ["yes"] * 4, ["yes", "yes", "yes", "no"], ["yes", "no", "no", "no"], random.Random(47), 500
        )
        self.assertAlmostEqual(out["value"], 0.5, places=4)
        self.assertTrue(out["brackets_zero"])

    def test_identical_label_sets_give_exactly_zero(self) -> None:
        pred = ["yes", "no", "maybe"]
        gold = ["yes", "yes", "maybe"]
        out = _paired_accuracy_delta(pred, gold, list(gold), random.Random(47), n_boot=100)
        self.assertEqual(out["value"], 0.0)
        self.assertEqual((out["ci_low"], out["ci_high"]), (0.0, 0.0))
        self.assertTrue(out["brackets_zero"])


class Rq8bSectionTests(unittest.TestCase):
    def _predictions_csv(self, path: Path, rows: list[dict]) -> Path:
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["method", "id", "gold", "predicted_label"])
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_joins_predictions_to_the_table_by_pmid_and_scores_both_label_sets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            preds = self._predictions_csv(
                Path(tmp) / "preds.csv",
                [
                    # Predicts the gold label on every item; the context-only annotator
                    # differs on pmids 3 and 4, so accuracy must drop against that label.
                    {"method": "m", "id": "pubmedqa-official-1", "gold": "yes", "predicted_label": "yes"},
                    {"method": "m", "id": "pubmedqa-official-3", "gold": "maybe", "predicted_label": "maybe"},
                    {"method": "m", "id": "pubmedqa-official-4", "gold": "maybe", "predicted_label": "maybe"},
                ],
            )
            out = rq8b_section(FIXTURE, preds, random.Random(47), n_boot=300)
        m = out["m"]
        self.assertEqual(m["n_cases"], 3)
        self.assertAlmostEqual(m["vs_final_decision"]["overall_accuracy"]["mean"], 1.0)
        self.assertAlmostEqual(m["vs_context_only_annotator"]["overall_accuracy"]["mean"], 1 / 3, places=4)
        self.assertAlmostEqual(m["accuracy_delta_context_only_minus_final"]["value"], -2 / 3, places=4)
        self.assertEqual(m["cases_where_the_two_golds_differ"], 2)

    def test_ceiling_is_computed_on_the_items_the_method_actually_ran(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            preds = self._predictions_csv(
                Path(tmp) / "preds.csv",
                # Only pmid 1, where the context-only annotator agrees with the gold, so
                # the ceiling on this one-item subset is 1.0 rather than the 0.6 overall.
                [{"method": "m", "id": "pubmedqa-official-1", "gold": "yes", "predicted_label": "no"}],
            )
            out = rq8b_section(FIXTURE, preds, random.Random(47), n_boot=200)
        m = out["m"]
        self.assertAlmostEqual(m["context_only_human_ceiling_on_these_items"]["mean"], 1.0)
        self.assertAlmostEqual(m["gap_to_ceiling"], -1.0)

    def test_unknown_pmids_and_blank_predictions_are_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            preds = self._predictions_csv(
                Path(tmp) / "preds.csv",
                [
                    {"method": "m", "id": "pubmedqa-official-1", "gold": "yes", "predicted_label": "yes"},
                    {"method": "m", "id": "pubmedqa-official-99999", "gold": "yes", "predicted_label": "yes"},
                    {"method": "m", "id": "pubmedqa-official-2", "gold": "no", "predicted_label": ""},
                ],
            )
            out = rq8b_section(FIXTURE, preds, random.Random(47), n_boot=100)
        self.assertEqual(out["m"]["n_cases"], 1)

    def test_missing_prediction_file_is_reported_not_raised(self) -> None:
        out = rq8b_section(FIXTURE, Path("/nonexistent/preds.csv"), random.Random(47), n_boot=10)
        self.assertIn("skipped", out)


class LoadTableTests(unittest.TestCase):
    def test_split_filter_and_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "table.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps(r)
                    for r in [_row("1", "yes", "yes", "yes"), _row("2", "no", "no", "no", split="train_dev")]
                ),
                encoding="utf-8",
            )
            self.assertEqual(len(load_table(path, None)), 2)
            self.assertEqual(len(load_table(path, "test")), 1)
            with self.assertRaises(SystemExit):
                load_table(Path(tmp) / "absent.jsonl", None)


if __name__ == "__main__":
    unittest.main()
