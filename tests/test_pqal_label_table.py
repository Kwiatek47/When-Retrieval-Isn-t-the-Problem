"""Tests for the PQA-L feature table and the bootstrap estimators H1 runs on.

Everything here works on synthetic items, so the suite never needs the 2.5 MB
`ori_pqal.json` (which is gitignored and fetched on demand).
"""

from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from scripts.agents.bootstrap_stats import (
    auroc,
    auroc_pairwise,
    bootstrap_auroc_ci,
    paired_auroc_delta_across_targets,
    paired_auroc_delta_ci,
    percentile,
)
from scripts.agents.build_pqal_label_table import (
    HEDGE_CUES,
    agreement_pattern,
    build_row,
    summarize,
    text_features,
)
from scripts.agents.pqal_official import (
    CONTEXT_ONLY_FIELD,
    GOLD_FIELD,
    SEES_CONCLUSION_FIELD,
    assert_annotator_roles,
)


def _item(**overrides) -> dict:
    item = {
        "QUESTION": "Is anorectal endosonography valuable in dyschesia?",
        "CONTEXTS": [
            "Dyschesia can be provoked by inappropriate defecation.",
            "The results showed a significant difference (p < 0.001) in 42 patients.",
        ],
        "LABELS": ["BACKGROUND", "RESULTS"],
        "MESHES": ["Adult", "Female"],
        "YEAR": "2002",
        CONTEXT_ONLY_FIELD: "yes",
        SEES_CONCLUSION_FIELD: "yes",
        GOLD_FIELD: "yes",
        "LONG_ANSWER": "Anorectal endosonography may be valuable, but further studies are needed.",
    }
    item.update(overrides)
    return item


class TextFeatureTests(unittest.TestCase):
    def test_hedge_count_finds_single_words_and_phrases(self) -> None:
        f = text_features("The effect may be present; further studies are needed.", "x")
        # "may", "may be", "further studies" all fire — the count is cues, not spans.
        self.assertEqual(f["x_hedge_count"], 3)
        self.assertGreater(f["x_hedge_density"], 0.0)

    def test_hedge_cues_match_on_word_boundaries(self) -> None:
        # "mayonnaise" must not count as the modal "may"; "MAY" must.
        self.assertEqual(text_features("mayonnaise dismayed maybe", "x")["x_hedge_count"], 0)
        self.assertEqual(text_features("It MAY happen", "x")["x_hedge_count"], 1)

    def test_empty_text_has_zero_density_not_nan(self) -> None:
        f = text_features("", "x")
        self.assertEqual(f["x_tokens"], 0)
        self.assertEqual(f["x_hedge_density"], 0.0)

    def test_pvalue_and_number_counts(self) -> None:
        f = text_features("Risk fell (p < 0.001) in 42 of 100 patients, P = .04.", "x")
        self.assertEqual(f["x_pvalue_count"], 2)  # "p < 0.001" and "P = .04"
        # 0.001, 42, 100, 04 — a leading-dot number counts as its digits, which is fine
        # for a magnitude-free count feature.
        self.assertEqual(f["x_number_count"], 4)

    def test_negative_result_cues(self) -> None:
        f = text_features("There was no significant difference; groups did not differ.", "x")
        self.assertGreaterEqual(f["x_negative_result_count"], 2)

    def test_hedge_lexicon_is_lowercase_and_deduplicated(self) -> None:
        self.assertEqual(len(HEDGE_CUES), len(set(HEDGE_CUES)))
        self.assertTrue(all(cue == cue.lower().strip() for cue in HEDGE_CUES))


class AgreementPatternTests(unittest.TestCase):
    def test_agreement(self) -> None:
        self.assertEqual(agreement_pattern("yes", "yes", "yes"), "agree")

    def test_dispute_resolved_towards_each_annotator(self) -> None:
        self.assertEqual(
            agreement_pattern("maybe", "yes", "maybe"),
            "dispute_final_follows_conclusion_annotator",
        )
        self.assertEqual(
            agreement_pattern("yes", "yes", "maybe"),
            "dispute_final_follows_context_only_annotator",
        )
        self.assertEqual(
            agreement_pattern("maybe", "yes", "no"),
            "dispute_final_follows_neither",
        )


class BuildRowTests(unittest.TestCase):
    def test_results_features_use_only_the_results_section(self) -> None:
        row = build_row("123", _item(), test_set={"123"})
        self.assertEqual(row["split"], "test")
        self.assertTrue(row["has_results_section"])
        self.assertEqual(row["results_pvalue_count"], 1)
        # The p-value lives in RESULTS, so the RESULTS span is shorter than the whole body.
        self.assertLess(row["results_tokens"], row["context_tokens"])

    def test_conclusion_features_come_from_long_answer(self) -> None:
        row = build_row("123", _item(), test_set=set())
        self.assertEqual(row["split"], "train_dev")
        self.assertGreater(row["conclusion_hedge_count"], 0)
        self.assertEqual(row["conclusion_tokens"], len(_item()["LONG_ANSWER"].split()) - 1 + 1)

    def test_item_without_results_section_is_flagged_and_scores_zero(self) -> None:
        row = build_row("9", _item(LABELS=["BACKGROUND", "CASE REPORTS"]), test_set=set())
        self.assertFalse(row["has_results_section"])
        self.assertEqual(row["results_tokens"], 0)
        self.assertEqual(row["results_hedge_density"], 0.0)

    def test_findings_counts_as_a_results_section(self) -> None:
        row = build_row("9", _item(LABELS=["BACKGROUND", "FINDINGS"]), test_set=set())
        self.assertTrue(row["has_results_section"])

    def test_row_is_json_serializable(self) -> None:
        json.dumps(build_row("123", _item(), test_set=set()))

    def test_summarize_counts_gold_maybe_annotator_patterns(self) -> None:
        rows = [
            build_row("1", _item(**{GOLD_FIELD: "maybe", SEES_CONCLUSION_FIELD: "maybe"}), set()),
            build_row("2", _item(**{GOLD_FIELD: "maybe", CONTEXT_ONLY_FIELD: "maybe"}), set()),
            build_row("3", _item(), set()),
        ]
        summary = summarize(rows)
        self.assertEqual(summary["n"], 3)
        self.assertEqual(summary["gold"]["maybe"], 2)
        self.assertEqual(sum(summary["gold_maybe_by_annotator_pattern"].values()), 2)


class AnnotatorRoleGuardTests(unittest.TestCase):
    def test_raises_when_the_two_annotator_fields_are_swapped(self) -> None:
        # Context-only annotator right twice, conclusion-reader right once: impossible
        # under the documented protocol, so the guard must fire.
        ori = {
            "1": {GOLD_FIELD: "yes", CONTEXT_ONLY_FIELD: "yes", SEES_CONCLUSION_FIELD: "no"},
            "2": {GOLD_FIELD: "no", CONTEXT_ONLY_FIELD: "no", SEES_CONCLUSION_FIELD: "no"},
        }
        with self.assertRaises(AssertionError):
            assert_annotator_roles(ori)

    def test_passes_and_reports_agreement_in_the_documented_direction(self) -> None:
        ori = {
            "1": {GOLD_FIELD: "yes", CONTEXT_ONLY_FIELD: "no", SEES_CONCLUSION_FIELD: "yes"},
            "2": {GOLD_FIELD: "no", CONTEXT_ONLY_FIELD: "no", SEES_CONCLUSION_FIELD: "no"},
        }
        agreement = assert_annotator_roles(ori)
        self.assertEqual(agreement[SEES_CONCLUSION_FIELD], 1.0)
        self.assertEqual(agreement[CONTEXT_ONLY_FIELD], 0.5)


class BootstrapStatsTests(unittest.TestCase):
    def test_auroc_perfect_and_inverted_and_tied(self) -> None:
        self.assertEqual(auroc([3.0, 4.0], [1.0, 2.0]), 1.0)
        self.assertEqual(auroc([1.0, 2.0], [3.0, 4.0]), 0.0)
        self.assertEqual(auroc([1.0, 1.0], [1.0, 1.0]), 0.5)  # all ties -> chance

    def test_auroc_empty_side_is_nan(self) -> None:
        self.assertNotEqual(auroc([], [1.0]), auroc([], [1.0]))  # NaN != NaN
        self.assertNotEqual(auroc_pairwise([1.0], []), auroc_pairwise([1.0], []))

    def test_fast_auroc_equals_the_pairwise_definition(self) -> None:
        """The rank-based AUROC must reproduce the pairwise one, ties included."""
        rng = random.Random(47)
        for _ in range(40):
            n_pos = rng.randint(1, 25)
            n_neg = rng.randint(1, 25)
            # Draw from a tiny discrete range so ties are frequent, not incidental.
            pos = [float(rng.randint(0, 4)) for _ in range(n_pos)]
            neg = [float(rng.randint(0, 4)) for _ in range(n_neg)]
            self.assertAlmostEqual(auroc(pos, neg), auroc_pairwise(pos, neg), places=12)

    def test_fast_auroc_handles_a_fully_tied_block_in_the_middle(self) -> None:
        pos = [1.0, 2.0, 2.0, 3.0]
        neg = [2.0, 2.0, 0.0]
        self.assertAlmostEqual(auroc(pos, neg), auroc_pairwise(pos, neg), places=12)

    def test_percentile_interpolates(self) -> None:
        self.assertEqual(percentile([0.0, 1.0], 0.5), 0.5)
        self.assertEqual(percentile([1.0, 2.0, 3.0], 0.0), 1.0)
        self.assertEqual(percentile([1.0, 2.0, 3.0], 1.0), 3.0)

    def test_bootstrap_ci_brackets_chance_for_a_useless_predictor(self) -> None:
        rng = random.Random(47)
        pos = [rng.gauss(0, 1) for _ in range(60)]
        neg = [rng.gauss(0, 1) for _ in range(60)]
        ci = bootstrap_auroc_ci(pos, neg, random.Random(47), n_boot=400)
        self.assertTrue(ci["brackets_chance"])
        self.assertEqual((ci["n_pos"], ci["n_neg"]), (60, 60))

    def test_paired_delta_detects_a_genuinely_better_predictor(self) -> None:
        rng = random.Random(47)
        labels = [i % 2 == 0 for i in range(200)]
        strong = [(1.0 if y else 0.0) + rng.gauss(0, 0.3) for y in labels]
        noise = [rng.gauss(0, 1) for _ in labels]
        out = paired_auroc_delta_ci(
            labels,
            {"strong": strong, "noise": noise},
            reference="strong",
            comparison="noise",
            rng=random.Random(47),
            n_boot=400,
        )
        self.assertGreater(out["delta"]["value"], 0.2)
        self.assertFalse(out["delta"]["brackets_zero"])
        self.assertFalse(out["per_predictor"]["strong"]["brackets_chance"])
        self.assertTrue(out["per_predictor"]["noise"]["brackets_chance"])

    def test_paired_delta_of_identical_predictors_is_exactly_zero(self) -> None:
        rng = random.Random(47)
        labels = [i % 3 == 0 for i in range(90)]
        scores = [rng.gauss(0, 1) for _ in labels]
        out = paired_auroc_delta_ci(
            labels,
            {"a": scores, "b": list(scores)},
            reference="a",
            comparison="b",
            rng=random.Random(47),
            n_boot=200,
        )
        self.assertEqual(out["delta"]["value"], 0.0)
        self.assertEqual((out["delta"]["ci_low"], out["delta"]["ci_high"]), (0.0, 0.0))

    def test_paired_delta_rejects_mismatched_lengths_and_unknown_names(self) -> None:
        with self.assertRaises(ValueError):
            paired_auroc_delta_ci([True, False], {"a": [1.0]}, "a", "a", random.Random(1), 10)
        with self.assertRaises(KeyError):
            paired_auroc_delta_ci([True, False], {"a": [1.0, 2.0]}, "a", "missing", random.Random(1), 10)

    def test_across_targets_detects_which_label_set_the_predictor_tracks(self) -> None:
        rng = random.Random(47)
        n = 300
        tracked = [i % 4 == 0 for i in range(n)]
        unrelated = [rng.random() < 0.25 for _ in range(n)]
        scores = [(1.0 if y else 0.0) + rng.gauss(0, 0.2) for y in tracked]
        out = paired_auroc_delta_across_targets(
            scores,
            {"tracked": tracked, "unrelated": unrelated},
            reference="tracked",
            comparison="unrelated",
            rng=random.Random(47),
            n_boot=300,
        )
        self.assertGreater(out["delta"]["value"], 0.2)
        self.assertFalse(out["delta"]["brackets_zero"])
        self.assertTrue(out["per_target"]["unrelated"]["brackets_chance"])
        self.assertEqual(out["per_target"]["tracked"]["n_pos"], sum(tracked))


class TableRoundTripTests(unittest.TestCase):
    def test_jsonl_rows_reload_with_the_fields_the_analyses_read(self) -> None:
        required = {
            "pmid", "split", "gold", "gold_is_maybe", "context_only_pred", "sees_conclusion_pred",
            "agreement_pattern", "has_results_section", "conclusion_hedge_density",
            "results_hedge_density", "context_hedge_density", "conclusion_tokens",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "table.jsonl"
            row = build_row("123", _item(), set())
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            reloaded = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        self.assertTrue(required <= set(reloaded))


if __name__ == "__main__":
    unittest.main()
