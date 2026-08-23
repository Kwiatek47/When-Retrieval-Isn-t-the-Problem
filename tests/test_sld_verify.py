"""Citation-gate tests: unknown sentence_ids and lexically ungrounded claims
must be dropped, and grounding_score/dropped_claims must reflect that."""

from __future__ import annotations

import unittest

from app.agents.sld.ledger import (
    Claim,
    ConclusionReconstructorContribution,
    EvidenceLedger,
    FindingsAuditorContribution,
    Gap,
    GapAuditorContribution,
    LedgerConflict,
    NeutralContribution,
    QuestionFramerContribution,
    RoundTwoOpinion,
)
from app.agents.sld.verify import (
    dropped_claims,
    grounding_score,
    verify_contribution,
    verify_ledger,
)

SENTENCES = {
    "S1": "Dyschesia can be provoked by inappropriate defecation movements.",
    "S2": "The aim of this prospective study was to demonstrate sphincter dysfunction.",
    "S3": "Twenty consecutive patients with dyschesia and twenty healthy controls were recruited.",
    "S4": "The anal sphincter became paradoxically thicker during straining in 85% of patients.",
    "S5": "Changes in sphincter length were statistically significantly different (p<0.01).",
}
SECTION_TAGS = {
    "S1": "OTHER",
    "S2": "BACKGROUND",
    "S3": "METHODS",
    "S4": "RESULTS",
    "S5": "RESULTS",
}


class TokenCoverageGateTests(unittest.TestCase):
    def test_claim_grounded_in_its_citation_survives(self) -> None:
        contribution = QuestionFramerContribution(
            agent_id="question_framer",
            target_population=Claim(
                text="twenty patients with dyschesia and twenty controls", sentence_ids=["S3"]
            ),
            target_exposure=Claim(
                text="twenty consecutive patients and twenty healthy controls recruited",
                sentence_ids=["S3"],
            ),
            target_outcome=Claim(text="sphincter dysfunction", sentence_ids=["S2"]),
            question_type="utility",
            yes_requires="a measurable difference is found",
            no_requires="no measurable difference is found",
        )
        result = verify_contribution(contribution, SENTENCES, section_tags=SECTION_TAGS)
        self.assertEqual(dropped_claims(result), [])
        self.assertEqual(grounding_score(result), 1.0)
        self.assertIsNotNone(result.value.target_population)

    def test_unknown_sentence_id_is_dropped(self) -> None:
        contribution = QuestionFramerContribution(
            agent_id="question_framer",
            target_population=Claim(text="twenty patients", sentence_ids=["S99"]),
            target_exposure=Claim(text="endosonography", sentence_ids=["S3"]),
            target_outcome=Claim(text="sphincter dysfunction", sentence_ids=["S2"]),
            question_type="utility",
            yes_requires="x",
            no_requires="y",
        )
        result = verify_contribution(contribution, SENTENCES, section_tags=SECTION_TAGS)
        self.assertIsNone(result.value.target_population)
        self.assertTrue(any("unknown sentence_ids" in d for d in dropped_claims(result)))
        self.assertLess(grounding_score(result), 1.0)

    def test_fabricated_claim_with_zero_word_overlap_is_dropped(self) -> None:
        """The citation exists, but the claim text shares no vocabulary with it —
        this is the hallucination case the token-coverage gate exists for."""
        contribution = FindingsAuditorContribution(
            agent_id="findings_auditor",
            primary_endpoint=Claim(
                text="patients reported improved quality of sleep after surgery",
                sentence_ids=["S4"],
            ),
            direction="positive",
            significance=Claim(text="p<0.01 significance reached", sentence_ids=["S5"]),
            effect_magnitude=Claim(text="85% of patients affected", sentence_ids=["S4"]),
        )
        result = verify_contribution(contribution, SENTENCES, section_tags=SECTION_TAGS)
        self.assertIsNone(result.value.primary_endpoint)
        self.assertIsNotNone(result.value.significance)
        self.assertIsNotNone(result.value.effect_magnitude)
        self.assertEqual(len(dropped_claims(result)), 1)
        self.assertAlmostEqual(grounding_score(result), 2 / 3)

    def test_findings_auditor_cannot_cite_outside_results(self) -> None:
        contribution = FindingsAuditorContribution(
            agent_id="findings_auditor",
            primary_endpoint=Claim(
                text="prospective study demonstrate sphincter dysfunction", sentence_ids=["S2"]
            ),
            direction="positive",
            significance=Claim(text="p<0.01 significance reached", sentence_ids=["S5"]),
            effect_magnitude=Claim(text="85% of patients affected", sentence_ids=["S4"]),
        )
        result = verify_contribution(contribution, SENTENCES, section_tags=SECTION_TAGS)
        self.assertIsNone(result.value.primary_endpoint)
        self.assertTrue(any("out-of-section" in d for d in dropped_claims(result)))

    def test_claim_with_no_citations_is_dropped(self) -> None:
        contribution = ConclusionReconstructorContribution(
            agent_id="conclusion_reconstructor",
            reconstructed_conclusion=Claim(text="endosonography is valuable", sentence_ids=[]),
            direction="positive",
            strength="qualified",
        )
        result = verify_contribution(contribution, SENTENCES)
        self.assertIsNone(result.value.reconstructed_conclusion)
        self.assertTrue(any("no sentence_ids" in d for d in dropped_claims(result)))

    def test_absent_claim_is_not_checked_or_dropped(self) -> None:
        contribution = QuestionFramerContribution(
            agent_id="question_framer",
            target_population=None,
            target_exposure=Claim(text="twenty consecutive patients recruited", sentence_ids=["S3"]),
            target_outcome=Claim(text="sphincter dysfunction", sentence_ids=["S2"]),
            question_type="utility",
            yes_requires="x",
            no_requires="y",
        )
        result = verify_contribution(contribution, SENTENCES, section_tags=SECTION_TAGS)
        self.assertNotIn("target_population", result.checked)
        self.assertEqual(grounding_score(result), 1.0)


class NeutralContributionVerificationTests(unittest.TestCase):
    def _neutral(self, **overrides) -> NeutralContribution:
        base = dict(
            agent_id="neutral_1",
            question_type="utility",
            direction="none",
            conclusion_direction="none",
            conclusion_strength="speculative",
        )
        base.update(overrides)
        return NeutralContribution(**base)

    def test_grounded_fields_all_survive(self) -> None:
        contribution = self._neutral(
            target_population=Claim(
                text="twenty consecutive patients recruited", sentence_ids=["S3"]
            ),
            primary_endpoint=Claim(
                text="the anal sphincter became paradoxically thicker during straining",
                sentence_ids=["S4"],
            ),
            significance=Claim(text="p<0.01 was reached", sentence_ids=["S5"]),
        )
        result = verify_contribution(contribution, SENTENCES, section_tags=SECTION_TAGS)
        self.assertEqual(dropped_claims(result), [])
        self.assertIsNotNone(result.value.target_population)
        self.assertIsNotNone(result.value.primary_endpoint)

    def test_primary_endpoint_cannot_cite_outside_results_even_for_neutral(self) -> None:
        contribution = self._neutral(
            primary_endpoint=Claim(
                text="prospective study demonstrate sphincter dysfunction", sentence_ids=["S2"]
            )
        )
        result = verify_contribution(contribution, SENTENCES, section_tags=SECTION_TAGS)
        self.assertIsNone(result.value.primary_endpoint)
        self.assertTrue(any("out-of-section" in d for d in dropped_claims(result)))

    def test_fabricated_gap_is_dropped(self) -> None:
        contribution = self._neutral(
            gaps=[
                Gap(
                    gap_type="underpowered",
                    description="Patients reported dizziness and nausea after the procedure.",
                    sentence_ids=["S3"],
                )
            ]
        )
        result = verify_contribution(contribution, SENTENCES, section_tags=SECTION_TAGS)
        self.assertEqual(result.value.gaps, [])


class GapAuditorVerificationTests(unittest.TestCase):
    def test_empty_gap_list_is_valid_and_fully_grounded(self) -> None:
        contribution = GapAuditorContribution(agent_id="gap_auditor", gaps=[])
        result = verify_contribution(contribution, SENTENCES, section_tags=SECTION_TAGS)
        self.assertEqual(result.value.gaps, [])
        self.assertEqual(grounding_score(result), 1.0)

    def test_gap_without_citation_is_allowed(self) -> None:
        gap = Gap(gap_type="no_comparator", description="No control group was used anywhere.")
        contribution = GapAuditorContribution(agent_id="gap_auditor", gaps=[gap])
        result = verify_contribution(contribution, SENTENCES, section_tags=SECTION_TAGS)
        self.assertEqual(len(result.value.gaps), 1)

    def test_gap_with_fabricated_citation_is_dropped(self) -> None:
        gap = Gap(
            gap_type="underpowered",
            description="Patients reported dizziness and nausea after the procedure.",
            sentence_ids=["S3"],
        )
        contribution = GapAuditorContribution(agent_id="gap_auditor", gaps=[gap])
        result = verify_contribution(contribution, SENTENCES, section_tags=SECTION_TAGS)
        self.assertEqual(result.value.gaps, [])
        self.assertEqual(len(dropped_claims(result)), 1)


class RoundTwoOpinionVerificationTests(unittest.TestCase):
    def test_grounded_rationale_survives(self) -> None:
        opinion = RoundTwoOpinion(
            agent_id="a",
            label="yes",
            rationale="Sphincter changes were statistically significantly different (p<0.01).",
            citations=["S5"],
        )
        result = verify_contribution(opinion, SENTENCES, section_tags=SECTION_TAGS)
        self.assertEqual(result.value.citations, ["S5"])
        self.assertEqual(dropped_claims(result), [])

    def test_unknown_citation_is_stripped_and_flagged(self) -> None:
        opinion = RoundTwoOpinion(
            agent_id="a",
            label="yes",
            rationale="Sphincter changes were statistically significantly different (p<0.01).",
            citations=["S5", "S99"],
        )
        result = verify_contribution(opinion, SENTENCES, section_tags=SECTION_TAGS)
        self.assertEqual(result.value.citations, ["S5"])
        self.assertTrue(any("unknown sentence_ids" in d for d in dropped_claims(result)))

    def test_fabricated_rationale_flagged_even_with_a_real_citation(self) -> None:
        opinion = RoundTwoOpinion(
            agent_id="a",
            label="yes",
            rationale="The medication caused severe liver toxicity in most patients.",
            citations=["S5"],
        )
        result = verify_contribution(opinion, SENTENCES, section_tags=SECTION_TAGS)
        self.assertTrue(any("low token coverage" in d for d in dropped_claims(result)))
        self.assertLess(grounding_score(result), 1.0)


class VerifyLedgerHallucinationTests(unittest.TestCase):
    """The scenario named explicitly in the design doc: a deliberately
    hallucinated ledger must come out of verify_ledger with its fabrications
    removed and a grounding_score that reflects the damage."""

    def test_deliberately_hallucinated_ledger_is_rejected(self) -> None:
        hallucinated = EvidenceLedger(
            target_population=Claim(
                text="elderly patients with heart failure recruited overseas",
                sentence_ids=["S3"],
            ),  # fabricated: no overlap with S3's actual content
            primary_endpoint=Claim(
                text="the anal sphincter became paradoxically thicker during straining",
                sentence_ids=["S4"],
            ),  # genuinely grounded
            significance=Claim(text="p<0.01 was reached", sentence_ids=["S5"]),  # grounded
            effect_magnitude=Claim(
                text="mortality dropped by half within a year", sentence_ids=["S99"]
            ),  # fabricated: unknown sentence id
            gaps=[
                Gap(
                    gap_type="underpowered",
                    description="Patients were followed for ten years post-surgery.",
                    sentence_ids=["S3"],
                )  # fabricated: not supported by S3
            ],
            conflicts=[
                LedgerConflict(
                    description="Fabricated disagreement about a drug interaction never mentioned.",
                    agent_ids=["findings_auditor"],
                    sentence_ids=["S1"],
                )
            ],
        )
        result = verify_ledger(hallucinated, SENTENCES)
        cleaned: EvidenceLedger = result.value

        # Fabrications removed.
        self.assertIsNone(cleaned.target_population)
        self.assertIsNone(cleaned.effect_magnitude)
        self.assertEqual(cleaned.gaps, [])
        self.assertEqual(cleaned.conflicts, [])

        # Genuinely grounded claims survive.
        self.assertIsNotNone(cleaned.primary_endpoint)
        self.assertIsNotNone(cleaned.significance)

        # The damage shows up quantitatively, not just structurally.
        self.assertGreaterEqual(len(dropped_claims(result)), 4)
        self.assertLess(grounding_score(result), 0.5)

    def test_fully_grounded_ledger_has_perfect_score(self) -> None:
        clean = EvidenceLedger(
            primary_endpoint=Claim(
                text="the anal sphincter became paradoxically thicker during straining",
                sentence_ids=["S4"],
            ),
            significance=Claim(text="p<0.01 was reached", sentence_ids=["S5"]),
        )
        result = verify_ledger(clean, SENTENCES)
        self.assertEqual(dropped_claims(result), [])
        self.assertEqual(grounding_score(result), 1.0)

    def test_empty_ledger_has_perfect_score_nothing_to_hallucinate(self) -> None:
        result = verify_ledger(EvidenceLedger(), SENTENCES)
        self.assertEqual(result.checked, [])
        self.assertEqual(grounding_score(result), 1.0)


if __name__ == "__main__":
    unittest.main()
