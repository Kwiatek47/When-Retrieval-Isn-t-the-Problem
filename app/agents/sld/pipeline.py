"""End-to-end orchestration: Stage 0 -> R1 panel -> gate #1 -> Moderator ->
verify_ledger -> R2 panel -> gate #2 -> Director -> compose_label.

This is the one place that wires every other module together; the ablation
ladder (design doc §7, L0-L9) is expressed entirely through
``SLDPipeline``'s constructor flags rather than separate code paths, so every
arm runs the same pipeline with different toggles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from app.agents.backends import InferenceBackend
from app.agents.sld import panel
from app.agents.sld.decision import (
    TriggerConfig,
    compose_label,
    fuse_with_biolinkbert,
    majority_vote_label,
    verdict_from_ledger,
)
from app.agents.sld.ledger import DirectorVerdict, PanelContribution, RoundTwoOpinion, SLDResult
from app.agents.sld.segmentation import (
    StatsProfile,
    classify_question_type,
    extract_abstract_text,
    extract_stats_profile,
    split_sentences,
    tag_sections,
)
from app.agents.sld.supervisor import LedgerSupervisor
from app.agents.sld.verify import verify_contribution

VerdictSource = Literal["rule", "llm"]


@dataclass
class SLDCase:
    case_id: str
    question: str
    abstract_raw: str
    expected_label: str | None = None
    biolinkbert_label: str | None = None


@dataclass
class SLDPipeline:
    backend: InferenceBackend

    personas: tuple[str, ...] = panel.R1_PERSONAS
    concurrency: int = 4

    r1_temperature: float = 0.3
    r1_num_predict: int | None = None
    r2_temperature: float = 0.3
    r2_num_predict: int | None = None
    moderator_temperature: float = 0.3
    moderator_num_predict: int | None = None
    director_temperature: float = 0.5
    director_num_predict: int | None = None
    director_samples: int = 3

    coverage_threshold: float = 0.5
    verdict_source: VerdictSource = "rule"
    trigger_config: TriggerConfig = field(default_factory=TriggerConfig)
    fuse_biolinkbert: bool = False

    # L3 (design doc §7): no round 2 at all — extraction + Moderator + rule
    # only, verdict derived heuristically from the ledger (decision.verdict_from_ledger).
    run_round_two: bool = True
    # L4 (design doc §7): round 2 runs, but agents see raw peer notes instead
    # of the verified ledger, and there is no Director call — final label is
    # a plain majority vote over round 2 labels (decision.majority_vote_label).
    show_ledger_in_r2: bool = True

    # Ablation (a) (design doc §7): verification always runs and is always
    # measured (grounding_score/dropped_claims stay populated either way) —
    # this only controls whether the *cleaned* or the *raw, unchecked* result
    # is what actually flows downstream into the ledger/prompts. Off = let
    # hallucinated claims through, to measure how much damage the gate
    # normally prevents.
    verification_enabled: bool = True
    # Ablation (b) (design doc §7): withhold the free, hallucination-proof
    # regex signal from R1 prompts, to measure its marginal value.
    use_stats_profile: bool = True

    async def run(self, case: SLDCase) -> SLDResult:
        abstract = extract_abstract_text(case.abstract_raw)
        sentence_list = split_sentences(abstract)
        sentences = dict(sentence_list)
        section_tags = tag_sections(sentence_list)
        stats_profile = extract_stats_profile(sentence_list) if self.use_stats_profile else StatsProfile()
        heuristic_question_type = classify_question_type(case.question)

        raw_r1 = await panel.run_round_one(
            question=case.question,
            sentences=sentences,
            section_tags=section_tags,
            stats_profile=stats_profile,
            backend=self.backend,
            personas=self.personas,
            concurrency=self.concurrency,
            temperature=self.r1_temperature,
            num_predict=self.r1_num_predict,
        )
        verified_r1: list[PanelContribution] = []
        r1_checked: list[str] = []
        r1_dropped: list[str] = []
        for contribution in raw_r1:
            result = verify_contribution(
                contribution,
                sentences,
                section_tags=section_tags,
                coverage_threshold=self.coverage_threshold,
            )
            verified_r1.append(result.value if self.verification_enabled else contribution)  # type: ignore[arg-type]
            r1_checked.extend(result.checked)
            r1_dropped.extend(result.dropped)
        grounding_r1 = 1.0 - (len(r1_dropped) / len(r1_checked)) if r1_checked else 1.0

        supervisor = LedgerSupervisor(
            backend=self.backend,
            temperature=self.moderator_temperature,
            num_predict=self.moderator_num_predict,
            coverage_threshold=self.coverage_threshold,
            verification_enabled=self.verification_enabled,
        )
        ledger = await supervisor.moderate(
            question=case.question,
            sentences=sentences,
            verified_r1=verified_r1,
            fallback_question_type=heuristic_question_type,
        )

        question_type = ledger.question_type or heuristic_question_type

        # L3: no round 2 at all. Verdict comes from the ledger heuristically;
        # compose_label still runs so the rule table stays the single source
        # of truth for label composition across every arm.
        if not self.run_round_two:
            verdict = verdict_from_ledger(ledger)
            label, rule_name = compose_label(verdict, question_type, self.trigger_config)
            if self.fuse_biolinkbert and case.biolinkbert_label is not None:
                label = fuse_with_biolinkbert(label, case.biolinkbert_label)
            return SLDResult(
                case_id=case.case_id,
                question=case.question,
                expected_label=case.expected_label,
                question_type=question_type,
                sentences=sentences,
                panel_r1_raw=raw_r1,
                panel_r1_verified=verified_r1,
                grounding_score_r1=grounding_r1,
                dropped_claims_r1=r1_dropped,
                ledger=ledger,
                director_verdict=verdict,
                predicted_label=label,
                rule_name=rule_name,
            )

        own_contributions = {contribution.agent_id: contribution for contribution in verified_r1}
        raw_r2 = await panel.run_round_two(
            question=case.question,
            sentences=sentences,
            ledger=ledger if self.show_ledger_in_r2 else None,
            own_contributions=own_contributions,
            round_instructions=ledger.round_instructions,
            backend=self.backend,
            concurrency=self.concurrency,
            temperature=self.r2_temperature,
            num_predict=self.r2_num_predict,
        )
        verified_r2: list[RoundTwoOpinion] = []
        r2_checked: list[str] = []
        r2_dropped: list[str] = []
        for opinion in raw_r2:
            result = verify_contribution(
                opinion, sentences, coverage_threshold=self.coverage_threshold
            )
            verified_r2.append(result.value if self.verification_enabled else opinion)  # type: ignore[arg-type]
            r2_checked.extend(result.checked)
            r2_dropped.extend(result.dropped)
        grounding_r2 = 1.0 - (len(r2_dropped) / len(r2_checked)) if r2_checked else 1.0

        # L4: round 2 ran without the ledger, so there is no verified shared
        # state for a Director to synthesize from — a plain majority vote is
        # the honest final step, not an LLM call dressed up as one.
        verdict: DirectorVerdict | None
        if not self.show_ledger_in_r2:
            label = majority_vote_label(verified_r2)
            rule_name = "majority_vote_no_ledger"
            verdict = None
        else:
            verdict = await supervisor.direct(
                question=case.question,
                ledger=ledger,
                round_two_opinions=verified_r2,
                samples=self.director_samples,
                temperature=self.director_temperature,
            )
            if self.verdict_source == "llm":
                label, rule_name = verdict.label, "director_llm_label"
            else:
                label, rule_name = compose_label(verdict, question_type, self.trigger_config)

        if self.fuse_biolinkbert and case.biolinkbert_label is not None:
            label = fuse_with_biolinkbert(label, case.biolinkbert_label)

        return SLDResult(
            case_id=case.case_id,
            question=case.question,
            expected_label=case.expected_label,
            question_type=question_type,
            sentences=sentences,
            panel_r1_raw=raw_r1,
            panel_r1_verified=verified_r1,
            grounding_score_r1=grounding_r1,
            dropped_claims_r1=r1_dropped,
            ledger=ledger,
            panel_r2_raw=raw_r2,
            panel_r2_verified=verified_r2,
            grounding_score_r2=grounding_r2,
            dropped_claims_r2=r2_dropped,
            director_verdict=verdict,
            predicted_label=label,
            rule_name=rule_name,
        )
