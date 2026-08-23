"""Deterministic label composition from the Director's structured verdict.

Design doc §4: this is the "rigor check recomputed in code" from the repo's
own `mas_architecture.md` direction, now applied to a typed verdict instead of
free text. Every ``maybe`` carries a named reason (``rule_name``), which is
what makes the decision layer ablatable — ``TriggerConfig`` lets each trigger
be disabled independently (design doc §7 ablation (e)).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal

from app.agents.sld.ledger import DirectorVerdict, EvidenceLedger, RoundTwoOpinion
from app.agents.sld.segmentation import QuestionType

Label = Literal["yes", "no", "maybe"]

# Only these question types make an unsupported finding decisive enough to
# answer "no" outright (design doc §4 decision table); other types fall
# through to the direction-based rules instead.
_NULL_RESULT_QUESTION_TYPES: frozenset[str] = frozenset({"causal", "comparison"})


@dataclass
class TriggerConfig:
    """Enable/disable individual ``maybe``/``no`` triggers for ablation."""

    coverage_gap_enabled: bool = True
    mixed_findings_enabled: bool = True
    null_result_enabled: bool = True
    hedged_conclusion_enabled: bool = True


def compose_label(
    verdict: DirectorVerdict,
    question_type: QuestionType | None,
    config: TriggerConfig | None = None,
) -> tuple[Label, str]:
    """Compose the final label from the Director's verdict.

    Returns ``(label, rule_name)``. Every branch — including the two
    direction-based ``yes``/``no`` outcomes — names its rule, so
    ``SLDResult.rule_name`` is always populated and every case is auditable,
    not just the ``maybe`` ones.
    """
    config = config or TriggerConfig()

    if config.coverage_gap_enabled and not verdict.question_answered_by_endpoint:
        return "maybe", "coverage_gap"
    if config.mixed_findings_enabled and not verdict.direction_determinate:
        return "maybe", "mixed_findings"
    if (
        config.null_result_enabled
        and not verdict.findings_statistically_supported
        and question_type in _NULL_RESULT_QUESTION_TYPES
    ):
        return "no", "null_result"
    if config.hedged_conclusion_enabled and verdict.conclusion_would_be_hedged:
        return "maybe", "hedged_conclusion"

    if verdict.direction == "positive":
        return "yes", "direction_positive"
    if verdict.direction == "negative":
        return "no", "direction_negative"
    return "maybe", "indeterminate"


def verdict_from_ledger(ledger: EvidenceLedger) -> DirectorVerdict:
    """Heuristic stand-in for the Director LLM call, derived purely from
    already-verified ledger fields.

    This is what the L3 arm (design doc §7: "R1 + Moderator + reguła, bez
    R2") uses instead of a Director call — it isolates the value of
    structured extraction + the rule table alone, with zero debate and zero
    Director LLM cost. ``direction_determinate`` is False specifically when
    findings_auditor's ``direction`` and conclusion_reconstructor's
    ``conclusion_direction`` disagree (a real, cheaply-detectable conflict),
    not just when direction is missing.
    """
    direction = ledger.direction or "none"
    conflicting = (
        ledger.conclusion_direction is not None
        and ledger.conclusion_direction != "none"
        and ledger.conclusion_direction != direction
    )
    if direction == "positive":
        label: Label = "yes"
    elif direction == "negative":
        label = "no"
    else:
        label = "maybe"
    citations: list[str] = []
    for claim in (ledger.primary_endpoint, ledger.significance, ledger.effect_magnitude):
        if claim is not None:
            citations.extend(claim.sentence_ids)
    return DirectorVerdict(
        question_answered_by_endpoint=ledger.primary_endpoint is not None,
        direction_determinate=direction != "none" and not conflicting,
        findings_statistically_supported=ledger.significance is not None,
        conclusion_would_be_hedged=ledger.conclusion_strength in ("qualified", "speculative"),
        direction=direction,  # type: ignore[arg-type]
        label=label,
        rationale="Heuristic verdict derived from the ledger (L3: no round 2, no Director call).",
        citations=citations,
    )


def majority_vote_label(opinions: list[RoundTwoOpinion]) -> Label:
    """0-LLM aggregation for the L4 arm (design doc §7: "R1 + R2, ale bez
    ledgera" — round 2 runs on raw peer notes instead of the verified ledger,
    and there is no Director call to synthesize a verdict from it). Ties
    resolve to the conservative ``maybe``, matching the rest of the decision
    layer's tie-breaking convention."""
    if not opinions:
        return "maybe"
    counts = Counter(opinion.label for opinion in opinions)
    top_count = max(counts.values())
    winners = [label for label, count in counts.items() if count == top_count]
    return winners[0] if len(winners) == 1 else "maybe"


def fuse_with_biolinkbert(label: Label, biolinkbert_label: str | None) -> Label:
    """``ledger_gate_fusion`` (design doc §4, the L8 arm).

    The rule-based label acts as a gate: ``maybe`` stays ``maybe`` (that's the
    class the rules are specifically designed to recover); any binary verdict
    is replaced by BioLinkBERT's binary label instead, since that classifier's
    yes/no accuracy is the harder baseline to beat.
    """
    if label == "maybe":
        return "maybe"
    if biolinkbert_label in ("yes", "no"):
        return biolinkbert_label  # type: ignore[return-value]
    return label
