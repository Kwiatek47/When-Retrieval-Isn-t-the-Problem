"""Tests for the RQ3 conclusiveness analysis (synthetic runs only)."""

from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from scripts.agents.analyze_rq3_conclusiveness import (
    _two_sample_delta,
    case_signal,
    contradiction_section,
    discover_runs,
    error_detector_section,
    load_run,
    maybe_detector_section,
    per_agent_section,
    qualitative_sample,
)


def _case(
    pmid: str,
    gold: str,
    predicted: str,
    verdicts: list[str | None],
    labels: list[str] | None = None,
    confidences: list[float] | None = None,
) -> dict:
    """One report case with `verdicts[i]` as agent i's evidence_conclusiveness."""
    labels = labels or [predicted] * len(verdicts)
    confidences = confidences if confidences is not None else [0.9] * len(verdicts)
    opinions = []
    for i, verdict in enumerate(verdicts):
        opinion = {"top_1_diagnosis": labels[i], "confidence_level": confidences[i]}
        if verdict is not None:
            opinion["evidence_conclusiveness"] = verdict
        opinions.append({"agent_id": f"agent_{i}", "persona": f"agent_{i}", "round": 2, "opinion": opinion})
    return {
        "id": f"pubmedqa-official-{pmid}",
        "expected_label": gold,
        "predicted_label": predicted,
        "rounds_run": 2,
        "aggregation_rule": "majority",
        "final_opinions": opinions,
    }


CONC, INC = "conclusive", "inconclusive"


class CaseSignalTests(unittest.TestCase):
    def test_fractions_and_flags(self) -> None:
        s = case_signal(_case("1", "maybe", "yes", [INC, CONC, INC, CONC]))
        self.assertEqual(s["pmid"], "1")
        self.assertEqual(s["n_inconclusive"], 2)
        self.assertAlmostEqual(s["inconclusive_fraction"], 0.5)
        self.assertTrue(s["any_inconclusive"])
        self.assertFalse(s["all_inconclusive"])
        self.assertTrue(s["committed_confidently"])
        self.assertFalse(s["correct"])
        self.assertEqual(s["agent_flags"], {"agent_0": 1.0, "agent_1": 0.0, "agent_2": 1.0, "agent_3": 0.0})

    def test_maybe_output_is_not_a_confident_commitment(self) -> None:
        s = case_signal(_case("1", "maybe", "maybe", [INC, INC]))
        self.assertFalse(s["committed_confidently"])
        self.assertTrue(s["correct"])
        self.assertTrue(s["all_inconclusive"])

    def test_blank_verdicts_make_the_case_unusable_rather_than_confident(self) -> None:
        # The mock backend writes "" instead of omitting the field. Reading that as
        # "conclusive" would invent unanimous confidence, so the case must drop out.
        self.assertIsNone(case_signal(_case("1", "yes", "yes", ["", "", "", ""])))
        self.assertIsNone(case_signal(_case("1", "yes", "yes", [None, None])))
        self.assertIsNone(case_signal(_case("1", "yes", "yes", ["unknown"])))

    def test_only_rated_agents_form_the_denominator(self) -> None:
        s = case_signal(_case("1", "yes", "yes", [INC, "", None, CONC]))
        self.assertEqual(s["n_agents"], 4)
        self.assertEqual(s["n_rated"], 2)
        self.assertAlmostEqual(s["inconclusive_fraction"], 0.5)
        self.assertEqual(sorted(s["agent_flags"]), ["agent_0", "agent_3"])

    def test_a_partially_rated_panel_can_still_be_unanimous(self) -> None:
        s = case_signal(_case("1", "yes", "yes", [INC, "", ""]))
        self.assertTrue(s["all_inconclusive"])

    def test_case_with_no_opinions_is_dropped(self) -> None:
        self.assertIsNone(case_signal({"id": "x", "expected_label": "yes", "final_opinions": []}))

    def test_missing_confidences_do_not_raise(self) -> None:
        case = _case("1", "yes", "yes", [INC, CONC])
        for o in case["final_opinions"]:
            o["opinion"].pop("confidence_level")
        s = case_signal(case)
        self.assertIsNone(s["mean_confidence"])
        self.assertIsNone(s["min_confidence"])


class LoadRunTests(unittest.TestCase):
    def _write(self, path: Path, payload: object) -> Path:
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_run_without_any_verdicts_is_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(Path(tmp) / "mock.json", {"cases": [_case("1", "yes", "yes", ["", ""])]})
            self.assertIsNone(load_run(p))

    def test_run_with_verdicts_returns_signals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(Path(tmp) / "real.json", {"cases": [_case("1", "yes", "yes", [INC, CONC])]})
            self.assertEqual(len(load_run(p)), 1)

    def test_malformed_and_shapeless_reports_are_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_text("{not json", encoding="utf-8")
            self.assertIsNone(load_run(bad))
            self.assertIsNone(load_run(self._write(Path(tmp) / "nocases.json", {"summary": {}})))

    def test_discover_runs_skips_prompt_dumps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("a.json", "a.prompts.json", "b.checkpoint.json", "b.json"):
                (root / name).write_text("{}", encoding="utf-8")
            self.assertEqual([p.name for p in discover_runs(root)], ["a.json", "b.json"])


# 4 cases: two flagged-and-committed (one of them gold maybe), one flagged answering maybe,
# one fully conclusive.
FIXTURE = [
    case_signal(_case("1", "maybe", "yes", [INC, CONC])),
    case_signal(_case("2", "yes", "no", [INC, INC])),
    case_signal(_case("3", "maybe", "maybe", [INC, CONC])),
    case_signal(_case("4", "yes", "yes", [CONC, CONC])),
]


class ContradictionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sec = contradiction_section(FIXTURE)

    def test_counts_match_a_hand_count(self) -> None:
        self.assertEqual(self.sec["n_cases"], 4)
        self.assertEqual(self.sec["n_any_inconclusive"], 3)
        self.assertEqual(self.sec["n_all_inconclusive"], 1)
        self.assertEqual(self.sec["n_committed_confidently_despite_flag"], 2)
        self.assertAlmostEqual(self.sec["share_of_flagged_that_still_commit"], 2 / 3, places=4)

    def test_gold_maybe_overridden_by_a_confident_answer_is_counted(self) -> None:
        self.assertEqual(self.sec["n_gold_maybe"], 2)
        self.assertEqual(self.sec["n_gold_maybe_flagged_but_committed"], 1)

    def test_emitted_labels_are_split_by_flag_level(self) -> None:
        by_flag = self.sec["emitted_by_flag"]
        self.assertEqual(by_flag["all_conclusive"], {"yes": 1})
        self.assertEqual(by_flag["all_inconclusive"], {"no": 1})
        self.assertEqual(by_flag["some_inconclusive"], {"yes": 1, "maybe": 1})

    def test_share_is_none_when_nothing_was_flagged(self) -> None:
        sec = contradiction_section([case_signal(_case("9", "yes", "yes", [CONC, CONC]))])
        self.assertIsNone(sec["share_of_flagged_that_still_commit"])


class PerAgentTests(unittest.TestCase):
    def test_a_constant_flag_gets_no_auroc_and_drives_the_spread(self) -> None:
        # agent_0 always flags, agent_1 never does — the pattern that makes "at least one
        # agent flagged it" a statement about the panel roster, not about the item.
        signals = [
            case_signal(_case("1", "maybe", "yes", [INC, CONC])),
            case_signal(_case("2", "yes", "yes", [INC, CONC])),
            case_signal(_case("3", "no", "no", [INC, CONC])),
        ]
        sec = per_agent_section(signals, random.Random(47), n_boot=200)
        self.assertAlmostEqual(sec["per_agent"]["agent_0"]["inconclusive_rate"]["mean"], 1.0)
        self.assertAlmostEqual(sec["per_agent"]["agent_1"]["inconclusive_rate"]["mean"], 0.0)
        self.assertAlmostEqual(sec["rate_spread"], 1.0)
        self.assertEqual(sec["max_rate_agent"], "agent_0")
        self.assertIsNone(sec["per_agent"]["agent_0"]["auroc_vs_gold_maybe"])
        self.assertIsNone(sec["per_agent"]["agent_1"]["auroc_vs_gold_maybe"])

    def test_a_varying_flag_gets_an_auroc(self) -> None:
        signals = [
            case_signal(_case("1", "maybe", "yes", [INC, CONC])),
            case_signal(_case("2", "yes", "yes", [CONC, CONC])),
            case_signal(_case("3", "no", "no", [CONC, CONC])),
        ]
        sec = per_agent_section(signals, random.Random(47), n_boot=200)
        auroc = sec["per_agent"]["agent_0"]["auroc_vs_gold_maybe"]
        self.assertIsNotNone(auroc)
        # Flag fires on the only gold `maybe` and on nothing else: a perfect separator.
        self.assertAlmostEqual(auroc["auroc"], 1.0)


class ErrorDetectorTests(unittest.TestCase):
    def test_accuracy_is_split_by_the_flag(self) -> None:
        sec = error_detector_section(FIXTURE, random.Random(47), n_boot=300)
        # flagged: pmids 1 (wrong), 2 (wrong), 3 (right) -> 1/3; clean: pmid 4 -> 1/1
        self.assertAlmostEqual(sec["accuracy_when_any_agent_inconclusive"]["mean"], 1 / 3, places=4)
        self.assertAlmostEqual(sec["accuracy_when_all_agents_conclusive"]["mean"], 1.0)
        self.assertAlmostEqual(sec["accuracy_delta_flagged_minus_clean"]["value"], 1 / 3 - 1.0, places=4)

    def test_an_arm_where_every_case_is_flagged_reports_no_delta(self) -> None:
        signals = [case_signal(_case("1", "yes", "yes", [INC, INC]))]
        sec = error_detector_section(signals, random.Random(47), n_boot=100)
        self.assertIsNone(sec["accuracy_when_all_agents_conclusive"])
        self.assertIsNone(sec["accuracy_delta_flagged_minus_clean"]["value"])


class MaybeDetectorTests(unittest.TestCase):
    def test_emitted_label_auroc_is_the_balanced_accuracy_of_its_maybe_decision(self) -> None:
        # 2 gold maybe (one emitted as maybe -> sens 0.5), 2 non-maybe (none emitted as
        # maybe -> spec 1.0); AUROC of a 0/1 indicator is (sens + spec) / 2 = 0.75.
        sec = maybe_detector_section(FIXTURE, random.Random(47), n_boot=300)
        self.assertAlmostEqual(sec["per_predictor"]["emitted_maybe"]["auroc"], 0.75)

    def test_confidence_predictor_appears_only_when_every_case_has_one(self) -> None:
        sec = maybe_detector_section(FIXTURE, random.Random(47), n_boot=200)
        self.assertIn("one_minus_mean_confidence", sec["per_predictor"])

        case = _case("5", "yes", "yes", [INC, CONC])
        for o in case["final_opinions"]:
            o["opinion"].pop("confidence_level")
        sec = maybe_detector_section(FIXTURE + [case_signal(case)], random.Random(47), n_boot=200)
        self.assertNotIn("one_minus_mean_confidence", sec["per_predictor"])

    def test_every_candidate_gets_a_paired_delta_against_the_emitted_label(self) -> None:
        sec = maybe_detector_section(FIXTURE, random.Random(47), n_boot=200)
        self.assertEqual(
            sorted(sec["deltas_vs_emitted"]),
            ["agent_maybe_fraction", "inconclusive_fraction", "one_minus_mean_confidence"],
        )
        for name, delta in sec["deltas_vs_emitted"].items():
            expected = sec["per_predictor"][name]["auroc"] - sec["per_predictor"]["emitted_maybe"]["auroc"]
            self.assertAlmostEqual(delta["value"], expected, places=4, msg=name)

    def test_arm_without_both_classes_is_skipped(self) -> None:
        signals = [case_signal(_case("1", "yes", "yes", [INC, CONC]))]
        self.assertIn("skipped", maybe_detector_section(signals, random.Random(47), n_boot=50))


class TwoSampleDeltaTests(unittest.TestCase):
    def test_sign_and_interval(self) -> None:
        out = _two_sample_delta([1.0] * 30, [0.0] * 30, random.Random(47), n_boot=400)
        self.assertAlmostEqual(out["value"], 1.0)
        self.assertFalse(out["brackets_zero"])

    def test_empty_group_is_reported_not_raised(self) -> None:
        out = _two_sample_delta([], [1.0], random.Random(47), n_boot=10)
        self.assertIsNone(out["value"])
        self.assertIn("reason", out)


class QualitativeSampleTests(unittest.TestCase):
    def test_selects_gold_maybe_that_was_flagged_and_overridden(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rows, _ = qualitative_sample({"armA": FIXTURE}, Path(tmp) / "absent.jsonl", size=10)
        # Only pmid 1 qualifies: gold maybe, flagged, answered yes. pmid 3 answered maybe.
        self.assertEqual([r["pmid"] for r in rows], ["1"])
        self.assertEqual(rows[0]["arm"], "armA")
        self.assertEqual(rows[0]["inconclusive_agents"], "1/2")

    def test_ranked_by_how_loudly_the_panel_objected(self) -> None:
        signals = [
            case_signal(_case("weak", "maybe", "yes", [INC, CONC, CONC, CONC])),
            case_signal(_case("loud", "maybe", "yes", [INC, INC, INC, INC])),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            rows, _ = qualitative_sample({"armA": signals}, Path(tmp) / "absent.jsonl", size=10)
        self.assertEqual([r["pmid"] for r in rows], ["loud", "weak"])

    def test_size_caps_the_sample(self) -> None:
        signals = [case_signal(_case(str(i), "maybe", "yes", [INC, INC])) for i in range(5)]
        with tempfile.TemporaryDirectory() as tmp:
            rows, _ = qualitative_sample({"armA": signals}, Path(tmp) / "absent.jsonl", size=2)
        self.assertEqual(len(rows), 2)

    def test_annotator_labels_are_joined_from_the_label_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            table = Path(tmp) / "table.jsonl"
            table.write_text(
                json.dumps({"pmid": "1", "context_only_pred": "yes", "sees_conclusion_pred": "maybe"}),
                encoding="utf-8",
            )
            rows, _ = qualitative_sample({"armA": FIXTURE}, table, size=10)
        self.assertEqual(rows[0]["context_only_annotator"], "yes")
        self.assertEqual(rows[0]["sees_conclusion_annotator"], "maybe")


if __name__ == "__main__":
    unittest.main()
