"""Schema tests for the typed Evidence Ledger models (Pydantic v2)."""

from __future__ import annotations

import unittest

from pydantic import TypeAdapter, ValidationError

from app.agents.sld.ledger import (
    Claim,
    ConclusionReconstructorContribution,
    DirectorVerdict,
    EvidenceLedger,
    FindingsAuditorContribution,
    Gap,
    GapAuditorContribution,
    LedgerConflict,
    PanelContribution,
    QuestionFramerContribution,
    RoundTwoOpinion,
    SLDResult,
)


class ClaimAndGapTests(unittest.TestCase):
    def test_claim_defaults_to_no_citations(self) -> None:
        claim = Claim(text="some assertion")
        self.assertEqual(claim.sentence_ids, [])

    def test_gap_requires_a_known_gap_type(self) -> None:
        with self.assertRaises(ValidationError):
            Gap(gap_type="not_a_real_type", description="x")
        gap = Gap(gap_type="underpowered", description="small n", sentence_ids=["S3"])
        self.assertEqual(gap.gap_type, "underpowered")


class PanelContributionUnionTests(unittest.TestCase):
    def test_discriminated_union_round_trips_each_persona(self) -> None:
        adapter = TypeAdapter(PanelContribution)
        contributions = [
            QuestionFramerContribution(
                agent_id="question_framer",
                target_population=Claim(text="adults", sentence_ids=["S1"]),
                target_exposure=Claim(text="drug X", sentence_ids=["S2"]),
                target_outcome=Claim(text="mortality", sentence_ids=["S2"]),
                question_type="utility",
                yes_requires="a measurable benefit",
                no_requires="no measurable benefit",
            ),
            FindingsAuditorContribution(
                agent_id="findings_auditor",
                primary_endpoint=Claim(text="mortality reduced", sentence_ids=["S6"]),
                direction="positive",
                significance=Claim(text="p<0.01", sentence_ids=["S7"]),
                effect_magnitude=Claim(text="85% vs 35%", sentence_ids=["S6"]),
            ),
            GapAuditorContribution(agent_id="gap_auditor", gaps=[]),
            ConclusionReconstructorContribution(
                agent_id="conclusion_reconstructor",
                reconstructed_conclusion=Claim(text="X is effective", sentence_ids=["S9"]),
                direction="positive",
                strength="qualified",
            ),
        ]
        for contribution in contributions:
            parsed = adapter.validate_python(contribution.model_dump())
            self.assertEqual(type(parsed), type(contribution))
            self.assertEqual(parsed.persona, contribution.persona)

    def test_empty_gaps_list_is_a_valid_answer(self) -> None:
        contribution = GapAuditorContribution(agent_id="gap_auditor", gaps=[])
        self.assertEqual(contribution.gaps, [])

    def test_findings_auditor_direction_is_constrained(self) -> None:
        with self.assertRaises(ValidationError):
            FindingsAuditorContribution(
                agent_id="findings_auditor",
                primary_endpoint=Claim(text="x"),
                direction="sideways",
                significance=Claim(text="x"),
                effect_magnitude=Claim(text="x"),
            )


class EvidenceLedgerTests(unittest.TestCase):
    def test_all_fields_optional_except_defaults(self) -> None:
        ledger = EvidenceLedger()
        self.assertIsNone(ledger.primary_endpoint)
        self.assertEqual(ledger.gaps, [])
        self.assertEqual(ledger.conflicts, [])
        self.assertEqual(ledger.round_instructions, [])

    def test_conflict_carries_agent_and_sentence_provenance(self) -> None:
        conflict = LedgerConflict(
            description="findings_auditor and conclusion_reconstructor disagree on direction",
            agent_ids=["findings_auditor", "conclusion_reconstructor"],
            sentence_ids=["S6", "S9"],
        )
        ledger = EvidenceLedger(conflicts=[conflict])
        self.assertEqual(len(ledger.conflicts), 1)
        self.assertEqual(ledger.conflicts[0].agent_ids, ["findings_auditor", "conclusion_reconstructor"])


class RoundTwoOpinionTests(unittest.TestCase):
    def test_label_is_constrained_to_yes_no_maybe(self) -> None:
        with self.assertRaises(ValidationError):
            RoundTwoOpinion(agent_id="a", label="unsure", rationale="x")

    def test_complement_and_self_audit_are_optional(self) -> None:
        opinion = RoundTwoOpinion(agent_id="a", label="yes", rationale="clear finding")
        self.assertIsNone(opinion.complement)
        self.assertIsNone(opinion.self_audit)


class DirectorVerdictTests(unittest.TestCase):
    def test_all_boolean_and_label_fields_required(self) -> None:
        with self.assertRaises(ValidationError):
            DirectorVerdict(label="yes")

    def test_valid_verdict_round_trips(self) -> None:
        verdict = DirectorVerdict(
            question_answered_by_endpoint=True,
            direction_determinate=True,
            findings_statistically_supported=True,
            conclusion_would_be_hedged=False,
            direction="positive",
            label="yes",
            rationale="clear positive finding",
            citations=["S6", "S7"],
        )
        self.assertEqual(DirectorVerdict.model_validate(verdict.model_dump()), verdict)


class SLDResultTests(unittest.TestCase):
    def test_minimal_result_only_needs_case_id_and_question(self) -> None:
        result = SLDResult(case_id="case-1", question="Is X valuable in Y?")
        self.assertIsNone(result.ledger)
        self.assertIsNone(result.director_verdict)
        self.assertIsNone(result.predicted_label)
        self.assertEqual(result.grounding_score_r1, 1.0)

    def test_panel_contributions_serialize_with_persona_discriminator(self) -> None:
        contribution = GapAuditorContribution(agent_id="gap_auditor", gaps=[])
        result = SLDResult(
            case_id="case-1",
            question="Is X valuable in Y?",
            panel_r1_raw=[contribution],
        )
        dumped = result.model_dump()
        self.assertEqual(dumped["panel_r1_raw"][0]["persona"], "gap_auditor")


if __name__ == "__main__":
    unittest.main()
