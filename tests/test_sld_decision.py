"""Tests for the deterministic label-composition rule table (decision.py)."""

from __future__ import annotations

import unittest

from app.agents.sld.decision import (
    TriggerConfig,
    compose_label,
    fuse_with_biolinkbert,
    majority_vote_label,
    verdict_from_ledger,
)
from app.agents.sld.ledger import Claim, DirectorVerdict, EvidenceLedger, RoundTwoOpinion


def _verdict(**overrides) -> DirectorVerdict:
    base = dict(
        question_answered_by_endpoint=True,
        direction_determinate=True,
        findings_statistically_supported=True,
        conclusion_would_be_hedged=False,
        direction="positive",
        label="yes",
        rationale="r",
        citations=[],
    )
    base.update(overrides)
    return DirectorVerdict(**base)


class ComposeLabelTests(unittest.TestCase):
    def test_coverage_gap_wins_first(self) -> None:
        verdict = _verdict(question_answered_by_endpoint=False)
        label, rule = compose_label(verdict, "utility")
        self.assertEqual((label, rule), ("maybe", "coverage_gap"))

    def test_mixed_findings_when_direction_not_determinate(self) -> None:
        verdict = _verdict(direction_determinate=False)
        label, rule = compose_label(verdict, "utility")
        self.assertEqual((label, rule), ("maybe", "mixed_findings"))

    def test_null_result_only_for_causal_or_comparison(self) -> None:
        verdict = _verdict(findings_statistically_supported=False)
        self.assertEqual(compose_label(verdict, "causal"), ("no", "null_result"))
        self.assertEqual(compose_label(verdict, "comparison"), ("no", "null_result"))
        # Same unsupported finding, but a question type where it doesn't
        # trigger "no" -> falls through to the direction-based rule.
        label, rule = compose_label(verdict, "utility")
        self.assertEqual((label, rule), ("yes", "direction_positive"))

    def test_hedged_conclusion(self) -> None:
        verdict = _verdict(conclusion_would_be_hedged=True)
        label, rule = compose_label(verdict, "utility")
        self.assertEqual((label, rule), ("maybe", "hedged_conclusion"))

    def test_direction_positive_yields_yes(self) -> None:
        verdict = _verdict(direction="positive")
        self.assertEqual(compose_label(verdict, "utility"), ("yes", "direction_positive"))

    def test_direction_negative_yields_no(self) -> None:
        verdict = _verdict(direction="negative")
        self.assertEqual(compose_label(verdict, "utility"), ("no", "direction_negative"))

    def test_direction_none_yields_maybe_indeterminate(self) -> None:
        verdict = _verdict(direction="none")
        self.assertEqual(compose_label(verdict, "utility"), ("maybe", "indeterminate"))

    def test_rule_priority_order(self) -> None:
        # Multiple triggers fire at once; coverage_gap must win over the rest.
        verdict = _verdict(
            question_answered_by_endpoint=False,
            direction_determinate=False,
            conclusion_would_be_hedged=True,
        )
        self.assertEqual(compose_label(verdict, "utility"), ("maybe", "coverage_gap"))

    def test_disabling_a_trigger_falls_through_to_the_next_rule(self) -> None:
        verdict = _verdict(question_answered_by_endpoint=False, direction="positive")
        config = TriggerConfig(coverage_gap_enabled=False)
        label, rule = compose_label(verdict, "utility", config)
        self.assertEqual((label, rule), ("yes", "direction_positive"))

    def test_disabling_every_trigger_falls_through_to_direction(self) -> None:
        verdict = _verdict(
            question_answered_by_endpoint=False,
            direction_determinate=False,
            findings_statistically_supported=False,
            conclusion_would_be_hedged=True,
            direction="negative",
        )
        config = TriggerConfig(
            coverage_gap_enabled=False,
            mixed_findings_enabled=False,
            null_result_enabled=False,
            hedged_conclusion_enabled=False,
        )
        self.assertEqual(compose_label(verdict, "causal", config), ("no", "direction_negative"))


class VerdictFromLedgerTests(unittest.TestCase):
    def test_positive_direction_with_endpoint_and_significance(self) -> None:
        ledger = EvidenceLedger(
            primary_endpoint=Claim(text="x improved", sentence_ids=["S1"]),
            direction="positive",
            significance=Claim(text="p<0.01", sentence_ids=["S2"]),
        )
        verdict = verdict_from_ledger(ledger)
        self.assertTrue(verdict.question_answered_by_endpoint)
        self.assertTrue(verdict.direction_determinate)
        self.assertTrue(verdict.findings_statistically_supported)
        self.assertEqual(verdict.direction, "positive")
        self.assertEqual(verdict.label, "yes")

    def test_missing_primary_endpoint_means_question_not_answered(self) -> None:
        verdict = verdict_from_ledger(EvidenceLedger())
        self.assertFalse(verdict.question_answered_by_endpoint)
        self.assertFalse(verdict.findings_statistically_supported)
        self.assertEqual(verdict.label, "maybe")

    def test_conflicting_directions_are_not_determinate(self) -> None:
        ledger = EvidenceLedger(direction="positive", conclusion_direction="negative")
        verdict = verdict_from_ledger(ledger)
        self.assertFalse(verdict.direction_determinate)

    def test_agreeing_directions_are_determinate(self) -> None:
        ledger = EvidenceLedger(direction="positive", conclusion_direction="positive")
        verdict = verdict_from_ledger(ledger)
        self.assertTrue(verdict.direction_determinate)

    def test_hedged_conclusion_strength_flags_hedging(self) -> None:
        self.assertTrue(
            verdict_from_ledger(EvidenceLedger(conclusion_strength="qualified")).conclusion_would_be_hedged
        )
        self.assertTrue(
            verdict_from_ledger(EvidenceLedger(conclusion_strength="speculative")).conclusion_would_be_hedged
        )
        self.assertFalse(
            verdict_from_ledger(EvidenceLedger(conclusion_strength="definitive")).conclusion_would_be_hedged
        )

    def test_citations_collected_from_findings_claims(self) -> None:
        ledger = EvidenceLedger(
            primary_endpoint=Claim(text="x", sentence_ids=["S1"]),
            significance=Claim(text="y", sentence_ids=["S2"]),
            effect_magnitude=Claim(text="z", sentence_ids=["S3"]),
        )
        verdict = verdict_from_ledger(ledger)
        self.assertEqual(set(verdict.citations), {"S1", "S2", "S3"})

    def test_feeds_cleanly_into_compose_label(self) -> None:
        ledger = EvidenceLedger(
            primary_endpoint=Claim(text="x", sentence_ids=["S1"]),
            direction="negative",
            significance=Claim(text="y", sentence_ids=["S2"]),
        )
        verdict = verdict_from_ledger(ledger)
        label, rule = compose_label(verdict, "utility")
        self.assertEqual((label, rule), ("no", "direction_negative"))


class MajorityVoteLabelTests(unittest.TestCase):
    def _opinion(self, label: str, agent_id: str = "a") -> RoundTwoOpinion:
        return RoundTwoOpinion(agent_id=agent_id, label=label, rationale="r")

    def test_clear_majority_wins(self) -> None:
        opinions = [self._opinion("yes"), self._opinion("yes"), self._opinion("no"), self._opinion("yes")]
        self.assertEqual(majority_vote_label(opinions), "yes")

    def test_tie_resolves_to_maybe(self) -> None:
        opinions = [self._opinion("yes"), self._opinion("no")]
        self.assertEqual(majority_vote_label(opinions), "maybe")

    def test_three_way_tie_resolves_to_maybe(self) -> None:
        opinions = [self._opinion("yes"), self._opinion("no"), self._opinion("maybe")]
        self.assertEqual(majority_vote_label(opinions), "maybe")

    def test_empty_list_is_maybe(self) -> None:
        self.assertEqual(majority_vote_label([]), "maybe")

    def test_unanimous(self) -> None:
        opinions = [self._opinion("no"), self._opinion("no")]
        self.assertEqual(majority_vote_label(opinions), "no")


class FuseWithBiolinkbertTests(unittest.TestCase):
    def test_maybe_always_stays_maybe(self) -> None:
        self.assertEqual(fuse_with_biolinkbert("maybe", "yes"), "maybe")
        self.assertEqual(fuse_with_biolinkbert("maybe", "no"), "maybe")
        self.assertEqual(fuse_with_biolinkbert("maybe", None), "maybe")

    def test_binary_label_replaced_by_biolinkbert(self) -> None:
        self.assertEqual(fuse_with_biolinkbert("yes", "no"), "no")
        self.assertEqual(fuse_with_biolinkbert("no", "yes"), "yes")

    def test_missing_biolinkbert_label_keeps_rule_label(self) -> None:
        self.assertEqual(fuse_with_biolinkbert("yes", None), "yes")
        self.assertEqual(fuse_with_biolinkbert("no", "maybe"), "no")


if __name__ == "__main__":
    unittest.main()
