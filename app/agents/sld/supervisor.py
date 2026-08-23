"""The Supervisor's two LLM-backed roles: Moderator (writes the ledger) and
Director (final structured verdict), plus the deterministic merge step that
keeps the Moderator's job narrow.

Design doc §4: "Blackboard MAS ... control component manages the board" is the
new role of the Supervisor here. Deliberately split in two:

- ``merge_verified_contributions`` (0 LLM) does the extraction — it just
  copies already citation-checked fields from Round 1 into a draft ledger.
- ``LedgerSupervisor.moderate`` (1 LLM call) does only judgment: conflicts,
  open questions, round instructions. It never re-asserts a fact the draft
  already has, so it cannot reintroduce a hallucination the verification gate
  already removed.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, field

from app.agents.backends import InferenceBackend
from app.agents.sld.ledger import (
    ConclusionReconstructorContribution,
    DirectorVerdict,
    EvidenceLedger,
    FindingsAuditorContribution,
    Gap,
    GapAuditorContribution,
    PanelContribution,
    QuestionFramerContribution,
    RoundTwoOpinion,
)
from app.agents.sld.panel import call_structured_llm
from app.agents.sld.prompts import (
    ModeratorSynthesis,
    build_director_prompt,
    build_moderator_prompt,
    render_ledger,
)
from app.agents.sld.segmentation import QuestionType
from app.agents.sld.verify import VerificationResult, verify_ledger


def merge_verified_contributions(
    verified_r1: list[PanelContribution],
    *,
    fallback_question_type: QuestionType | None = None,
) -> EvidenceLedger:
    """Deterministic, 0-LLM merge of already-verified Round 1 fields.

    Not "the Moderator's opinion" — this is just relocation of facts that were
    already extracted and citation-checked. Missing/dropped fields stay
    ``None``; a caller that wants a heuristic fallback for ``question_type``
    (e.g. from ``segmentation.classify_question_type``, if question_framer's
    own contribution was itself dropped) passes ``fallback_question_type``.
    """
    fields: dict[str, object] = {}
    gaps: list[Gap] = []

    for contribution in verified_r1:
        if isinstance(contribution, QuestionFramerContribution):
            fields["target_population"] = contribution.target_population
            fields["target_exposure"] = contribution.target_exposure
            fields["target_outcome"] = contribution.target_outcome
            fields["question_type"] = contribution.question_type
        elif isinstance(contribution, FindingsAuditorContribution):
            fields["primary_endpoint"] = contribution.primary_endpoint
            fields["direction"] = contribution.direction
            fields["significance"] = contribution.significance
            fields["effect_magnitude"] = contribution.effect_magnitude
        elif isinstance(contribution, GapAuditorContribution):
            gaps.extend(contribution.gaps)
        elif isinstance(contribution, ConclusionReconstructorContribution):
            fields["reconstructed_conclusion"] = contribution.reconstructed_conclusion
            fields["conclusion_direction"] = contribution.direction
            fields["conclusion_strength"] = contribution.strength

    if fields.get("question_type") is None and fallback_question_type is not None:
        fields["question_type"] = fallback_question_type

    return EvidenceLedger(gaps=gaps, **fields)  # type: ignore[arg-type]


def _fallback_moderator_synthesis() -> ModeratorSynthesis:
    return ModeratorSynthesis(
        conflicts=[],
        open_questions=[],
        round_instructions=["Round 1 moderation failed; answer conservatively from the ledger alone."],
    )


def _fallback_director_verdict() -> DirectorVerdict:
    return DirectorVerdict(
        question_answered_by_endpoint=False,
        direction_determinate=False,
        findings_statistically_supported=False,
        conclusion_would_be_hedged=True,
        direction="none",
        label="maybe",
        rationale="Fallback after empty/invalid model output; discount this verdict.",
        citations=[],
    )


def _majority_categorical(values: list[str], *, conservative: str) -> str:
    counts = Counter(values)
    if not counts:
        return conservative
    top_count = max(counts.values())
    winners = [value for value, count in counts.items() if count == top_count]
    return winners[0] if len(winners) == 1 else conservative


def _majority_bool(values: list[bool], *, tie_break: bool) -> bool:
    true_count = sum(values)
    false_count = len(values) - true_count
    if true_count > false_count:
        return True
    if false_count > true_count:
        return False
    return tie_break


def aggregate_director_verdicts(verdicts: list[DirectorVerdict]) -> DirectorVerdict:
    """Field-by-field self-consistency aggregation (design doc §4).

    Not "most common whole object" — with 4 independent booleans the odds of
    an exact full-object match collapse fast as sample count grows. Each
    boolean is a 2/3-style majority vote; ties on the cautious-vs-not booleans
    resolve toward the reading that leads compose_label() to hedge (``maybe``)
    rather than commit. ``direction``/``label`` ties resolve to the
    conservative value directly. ``rationale``/``citations`` are taken from
    whichever single sample agrees with the aggregate on the most fields
    (never concatenated — that would splice together contradictory reasoning).
    """
    if not verdicts:
        raise ValueError("aggregate_director_verdicts requires at least one verdict")
    if len(verdicts) == 1:
        return verdicts[0]

    question_answered = _majority_bool(
        [v.question_answered_by_endpoint for v in verdicts], tie_break=False
    )
    direction_determinate = _majority_bool(
        [v.direction_determinate for v in verdicts], tie_break=False
    )
    findings_supported = _majority_bool(
        [v.findings_statistically_supported for v in verdicts], tie_break=False
    )
    hedged = _majority_bool([v.conclusion_would_be_hedged for v in verdicts], tie_break=True)
    direction = _majority_categorical([v.direction for v in verdicts], conservative="none")
    label = _majority_categorical([v.label for v in verdicts], conservative="maybe")

    aggregated_key = (
        question_answered,
        direction_determinate,
        findings_supported,
        hedged,
        direction,
        label,
    )

    def _distance(v: DirectorVerdict) -> int:
        own_key = (
            v.question_answered_by_endpoint,
            v.direction_determinate,
            v.findings_statistically_supported,
            v.conclusion_would_be_hedged,
            v.direction,
            v.label,
        )
        return sum(1 for a, b in zip(own_key, aggregated_key) if a != b)

    closest = min(verdicts, key=_distance)

    return DirectorVerdict(
        question_answered_by_endpoint=question_answered,
        direction_determinate=direction_determinate,
        findings_statistically_supported=findings_supported,
        conclusion_would_be_hedged=hedged,
        direction=direction,  # type: ignore[arg-type]
        label=label,  # type: ignore[arg-type]
        rationale=closest.rationale,
        citations=list(closest.citations),
    )


@dataclass
class LedgerSupervisor:
    backend: InferenceBackend
    temperature: float = 0.3
    num_predict: int | None = None
    coverage_threshold: float = 0.5
    last_ledger_verification: VerificationResult | None = field(default=None, init=False)

    async def moderate(
        self,
        *,
        question: str,
        sentences: dict[str, str],
        verified_r1: list[PanelContribution],
        fallback_question_type: QuestionType | None = None,
    ) -> EvidenceLedger:
        draft = merge_verified_contributions(
            verified_r1, fallback_question_type=fallback_question_type
        )
        prompt = build_moderator_prompt(question, verified_r1)
        synthesis = await call_structured_llm(
            self.backend,
            user_prompt=prompt,
            model_cls=ModeratorSynthesis,
            fallback=_fallback_moderator_synthesis(),
            temperature=self.temperature,
            num_predict=self.num_predict,
            label="moderator",
        )
        candidate = draft.model_copy(
            update={
                "conflicts": synthesis.conflicts,
                "open_questions": synthesis.open_questions,
                "round_instructions": synthesis.round_instructions,
            }
        )
        result = verify_ledger(candidate, sentences, coverage_threshold=self.coverage_threshold)
        self.last_ledger_verification = result
        return result.value  # type: ignore[return-value]

    async def direct(
        self,
        *,
        question: str,
        sentences: dict[str, str],
        ledger: EvidenceLedger,
        round_two_opinions: list[RoundTwoOpinion],
        samples: int = 3,
        temperature: float = 0.5,
    ) -> DirectorVerdict:
        prompt = build_director_prompt(question, sentences, render_ledger(ledger), round_two_opinions)

        async def _one(index: int) -> DirectorVerdict:
            return await call_structured_llm(
                self.backend,
                user_prompt=prompt,
                model_cls=DirectorVerdict,
                fallback=_fallback_director_verdict(),
                temperature=temperature,
                num_predict=self.num_predict,
                label=f"director[{index}]",
            )

        verdicts = list(
            await asyncio.gather(*[_one(i) for i in range(max(1, samples))])
        )
        return aggregate_director_verdicts(verdicts)
