"""Tests for the deterministic parts of supervisor.py: the 0-LLM merge and
the field-wise Director self-consistency aggregation."""

from __future__ import annotations

import unittest

from app.agents.sld.ledger import (
    Claim,
    ConclusionReconstructorContribution,
    DirectorVerdict,
    FindingsAuditorContribution,
    Gap,
    GapAuditorContribution,
    NeutralContribution,
    QuestionFramerContribution,
)
from app.agents.sld.supervisor import (
    aggregate_director_verdicts,
    label_agreement_fraction,
    merge_neutral_contributions,
    merge_verified_contributions,
)


class MergeVerifiedContributionsTests(unittest.TestCase):
    def test_merges_all_four_personas_losslessly(self) -> None:
        contributions = [
            QuestionFramerContribution(
                agent_id="question_framer",
                target_population=Claim(text="patients", sentence_ids=["S1"]),
                target_exposure=Claim(text="drug X", sentence_ids=["S1"]),
                target_outcome=Claim(text="mortality", sentence_ids=["S1"]),
                question_type="utility",
                yes_requires="a measurable benefit",
                no_requires="no measurable benefit",
            ),
            FindingsAuditorContribution(
                agent_id="findings_auditor",
                primary_endpoint=Claim(text="mortality reduced", sentence_ids=["S2"]),
                direction="positive",
                significance=Claim(text="p<0.01", sentence_ids=["S2"]),
                effect_magnitude=Claim(text="85% vs 35%", sentence_ids=["S2"]),
            ),
            GapAuditorContribution(
                agent_id="gap_auditor",
                gaps=[Gap(gap_type="underpowered", description="small n", sentence_ids=["S1"])],
            ),
            ConclusionReconstructorContribution(
                agent_id="conclusion_reconstructor",
                reconstructed_conclusion=Claim(text="X is effective", sentence_ids=["S2"]),
                direction="positive",
                strength="qualified",
            ),
        ]
        ledger = merge_verified_contributions(contributions)
        self.assertEqual(ledger.target_population.text, "patients")
        self.assertEqual(ledger.question_type, "utility")
        self.assertEqual(ledger.primary_endpoint.text, "mortality reduced")
        self.assertEqual(ledger.direction, "positive")
        self.assertEqual(len(ledger.gaps), 1)
        self.assertEqual(ledger.gaps[0].gap_type, "underpowered")
        self.assertEqual(ledger.conclusion_direction, "positive")
        self.assertEqual(ledger.conclusion_strength, "qualified")
        # No LLM ran; conflicts/open_questions/round_instructions stay empty.
        self.assertEqual(ledger.conflicts, [])
        self.assertEqual(ledger.round_instructions, [])

    def test_missing_persona_leaves_its_fields_none(self) -> None:
        ledger = merge_verified_contributions(
            [FindingsAuditorContribution(agent_id="findings_auditor", direction="none")]
        )
        self.assertIsNone(ledger.target_population)
        self.assertIsNone(ledger.reconstructed_conclusion)
        self.assertEqual(ledger.direction, "none")  # findings_auditor's own field, unaffected

    def test_dropped_claim_field_stays_none_even_when_persona_present(self) -> None:
        # Simulates verify.py having nulled out primary_endpoint but kept the
        # contribution (direction is a plain field, not citation-bearing).
        ledger = merge_verified_contributions(
            [
                FindingsAuditorContribution(
                    agent_id="findings_auditor", primary_endpoint=None, direction="positive"
                )
            ]
        )
        self.assertIsNone(ledger.primary_endpoint)
        self.assertEqual(ledger.direction, "positive")

    def test_fallback_question_type_used_only_when_question_framer_absent(self) -> None:
        ledger = merge_verified_contributions(
            [FindingsAuditorContribution(agent_id="findings_auditor", direction="none")],
            fallback_question_type="causal",
        )
        self.assertEqual(ledger.question_type, "causal")

        ledger_with_framer = merge_verified_contributions(
            [
                QuestionFramerContribution(
                    agent_id="question_framer",
                    question_type="prevalence",
                    yes_requires="x",
                    no_requires="y",
                )
            ],
            fallback_question_type="causal",
        )
        self.assertEqual(ledger_with_framer.question_type, "prevalence")

    def test_empty_input_yields_empty_ledger(self) -> None:
        ledger = merge_verified_contributions([])
        self.assertIsNone(ledger.target_population)
        self.assertEqual(ledger.gaps, [])


def _verdict(**overrides) -> DirectorVerdict:
    base = dict(
        question_answered_by_endpoint=True,
        direction_determinate=True,
        findings_statistically_supported=True,
        conclusion_would_be_hedged=False,
        direction="positive",
        label="yes",
        rationale="default rationale",
        citations=["S1"],
    )
    base.update(overrides)
    return DirectorVerdict(**base)


class AggregateDirectorVerdictsTests(unittest.TestCase):
    def test_single_verdict_passes_through_unchanged(self) -> None:
        verdict = _verdict()
        self.assertEqual(aggregate_director_verdicts([verdict]), verdict)

    def test_unanimous_verdicts_aggregate_to_the_same_result(self) -> None:
        verdicts = [_verdict(rationale=f"r{i}") for i in range(3)]
        agg = aggregate_director_verdicts(verdicts)
        self.assertTrue(agg.question_answered_by_endpoint)
        self.assertEqual(agg.label, "yes")
        self.assertEqual(agg.direction, "positive")

    def test_two_of_three_majority_wins_and_rationale_comes_from_a_majority_sample(self) -> None:
        majority = _verdict(rationale="majority rationale")
        minority = _verdict(
            question_answered_by_endpoint=False,
            direction_determinate=False,
            findings_statistically_supported=False,
            conclusion_would_be_hedged=True,
            direction="none",
            label="maybe",
            rationale="minority rationale",
        )
        agg = aggregate_director_verdicts([majority, majority, minority])
        self.assertEqual(agg.label, "yes")
        self.assertTrue(agg.question_answered_by_endpoint)
        self.assertEqual(agg.rationale, "majority rationale")

    def test_boolean_tie_resolves_toward_the_cautious_reading(self) -> None:
        # question_answered_by_endpoint: 1 True, 1 False -> tie -> False (cautious)
        # conclusion_would_be_hedged: 1 True, 1 False -> tie -> True (cautious)
        a = _verdict(question_answered_by_endpoint=True, conclusion_would_be_hedged=False)
        b = _verdict(question_answered_by_endpoint=False, conclusion_would_be_hedged=True)
        agg = aggregate_director_verdicts([a, b])
        self.assertFalse(agg.question_answered_by_endpoint)
        self.assertTrue(agg.conclusion_would_be_hedged)

    def test_direction_and_label_ties_resolve_to_conservative_values(self) -> None:
        yes = _verdict(direction="positive", label="yes")
        no = _verdict(direction="negative", label="no")
        agg = aggregate_director_verdicts([yes, no])
        self.assertEqual(agg.direction, "none")
        self.assertEqual(agg.label, "maybe")

    def test_rationale_never_concatenates_samples(self) -> None:
        verdicts = [_verdict(rationale="alpha"), _verdict(rationale="beta"), _verdict(rationale="gamma")]
        agg = aggregate_director_verdicts(verdicts)
        self.assertIn(agg.rationale, {"alpha", "beta", "gamma"})
        self.assertNotIn(" ", agg.rationale.replace("alpha", "").replace("beta", "").replace("gamma", ""))

    def test_raises_on_empty_input(self) -> None:
        with self.assertRaises(ValueError):
            aggregate_director_verdicts([])


class MergeNeutralContributionsTests(unittest.TestCase):
    def _neutral(self, agent_id: str, **overrides) -> NeutralContribution:
        base = dict(
            agent_id=agent_id,
            question_type="utility",
            direction="none",
            conclusion_direction="none",
            conclusion_strength="speculative",
        )
        base.update(overrides)
        return NeutralContribution(**base)

    def test_categorical_fields_use_majority_vote(self) -> None:
        contributions = [
            self._neutral("neutral_1", direction="positive"),
            self._neutral("neutral_2", direction="positive"),
            self._neutral("neutral_3", direction="negative"),
        ]
        ledger = merge_neutral_contributions(contributions)
        self.assertEqual(ledger.direction, "positive")

    def test_categorical_tie_falls_back_to_conservative(self) -> None:
        contributions = [
            self._neutral("neutral_1", direction="positive"),
            self._neutral("neutral_2", direction="negative"),
        ]
        ledger = merge_neutral_contributions(contributions)
        self.assertEqual(ledger.direction, "none")

    def test_claim_fields_use_first_available(self) -> None:
        contributions = [
            self._neutral("neutral_1", primary_endpoint=None),
            self._neutral("neutral_2", primary_endpoint=Claim(text="x improved", sentence_ids=["S2"])),
            self._neutral("neutral_3", primary_endpoint=Claim(text="y worsened", sentence_ids=["S3"])),
        ]
        ledger = merge_neutral_contributions(contributions)
        self.assertEqual(ledger.primary_endpoint.text, "x improved")

    def test_gaps_are_unioned_not_deduplicated(self) -> None:
        gap = Gap(gap_type="underpowered", description="small n", sentence_ids=["S1"])
        contributions = [
            self._neutral("neutral_1", gaps=[gap]),
            self._neutral("neutral_2", gaps=[gap]),
            self._neutral("neutral_3", gaps=[]),
        ]
        ledger = merge_neutral_contributions(contributions)
        self.assertEqual(len(ledger.gaps), 2)

    def test_empty_input_yields_empty_ledger(self) -> None:
        ledger = merge_neutral_contributions([])
        self.assertIsNone(ledger.target_population)
        self.assertEqual(ledger.gaps, [])


class LabelAgreementFractionTests(unittest.TestCase):
    def test_single_sample_has_no_signal(self) -> None:
        self.assertIsNone(label_agreement_fraction([_verdict(label="yes")], "yes"))

    def test_unanimous_agreement_is_one(self) -> None:
        verdicts = [_verdict(label="yes") for _ in range(3)]
        self.assertEqual(label_agreement_fraction(verdicts, "yes"), 1.0)

    def test_partial_agreement_is_the_matching_fraction(self) -> None:
        verdicts = [_verdict(label="yes"), _verdict(label="yes"), _verdict(label="maybe")]
        self.assertAlmostEqual(label_agreement_fraction(verdicts, "yes"), 2 / 3)

    def test_no_sample_matching_aggregated_label_is_zero(self) -> None:
        verdicts = [_verdict(label="yes"), _verdict(label="no")]
        self.assertEqual(label_agreement_fraction(verdicts, "maybe"), 0.0)


if __name__ == "__main__":
    unittest.main()
