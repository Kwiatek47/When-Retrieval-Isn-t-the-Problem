"""Aggregate multi-agent ClinicalOpinion outputs into a PubMedQA label (no supervisor)."""

from __future__ import annotations

from collections import Counter
import json
import re
from typing import Any

from app.agents.models import AgentRoundOpinion, ClinicalOpinion, ConsensusDecision, RankedHypothesis
from app.agents.supervisor_agent import SupervisorAgent

_LABELS = ("yes", "no", "maybe")
_LABEL_RE = re.compile(r"\b(yes|no|maybe)\b", re.IGNORECASE)


def extract_label(text: str) -> str | None:
    """Map free text / top_1_diagnosis to yes|no|maybe."""
    normalized = (text or "").strip().lower()
    if normalized in _LABELS:
        return normalized
    match = _LABEL_RE.search(normalized)
    return match.group(1).lower() if match else None


def opinion_label(opinion: ClinicalOpinion) -> str | None:
    return extract_label(opinion.top_1_diagnosis)


def majority_vote(opinions: list[ClinicalOpinion]) -> tuple[str | None, dict[str, float]]:
    """
    Confidence-weighted majority over agent opinions.

    Tie-break: vote count, then confidence sum, then yes > no > maybe.
    Returns (label, vote_count_share_by_label).
    """
    scored: dict[str, float] = {label: 0.0 for label in _LABELS}
    counts: Counter[str] = Counter()
    for opinion in opinions:
        label = opinion_label(opinion)
        if label is None:
            continue
        counts[label] += 1
        scored[label] += float(opinion.confidence_level)

    if not counts:
        return None, scored

    order = {"yes": 0, "no": 1, "maybe": 2}
    best = max(
        counts.keys(),
        key=lambda label: (counts[label], scored[label], -order[label]),
    )
    total = sum(counts.values())
    share = {label: counts.get(label, 0) / total for label in _LABELS}
    return best, share


def weighted_vote(
    opinions: list[ClinicalOpinion],
    *,
    weights: list[float] | None = None,
) -> tuple[str | None, dict[str, float], dict[str, float]]:
    """Confidence × optional per-opinion weight majority."""
    scored: dict[str, float] = {label: 0.0 for label in _LABELS}
    counts: Counter[str] = Counter()
    for index, opinion in enumerate(opinions):
        label = opinion_label(opinion)
        if label is None:
            continue
        weight = 1.0 if weights is None else float(weights[index])
        counts[label] += 1
        scored[label] += float(opinion.confidence_level) * weight

    if not counts:
        return None, scored, scored

    order = {"yes": 0, "no": 1, "maybe": 2}
    best = max(
        counts.keys(),
        key=lambda label: (scored[label], counts[label], -order[label]),
    )
    total = sum(scored.values()) or 1.0
    share = {label: scored[label] / total for label in _LABELS}
    return best, share, scored


def aggregate_pubmedqa_decision(
    agent_opinions: list[ClinicalOpinion],
    *,
    bert_opinion: ClinicalOpinion | None = None,
    mode: str = "majority",
    bert_gate_confidence: float = 0.90,
    bert_vote_weight: float = 3.0,
) -> tuple[str | None, dict[str, float], str]:
    """
    Aggregate panel (+ optional BioLinkBERT) into a final yes/no/maybe.

    Modes:
      - majority: equal votes (confidence-weighted)
      - bert_weighted: BioLinkBERT vote amplified by bert_vote_weight
      - bert_gate: trust high-confidence BioLinkBERT yes/no unless the full panel
        unanimously disagrees; for maybe/low-confidence BERT, use weighted debate
    """
    mode = (mode or "majority").strip().lower()
    if bert_opinion is None or mode == "majority":
        opinions = [*agent_opinions, bert_opinion] if bert_opinion is not None else agent_opinions
        label, share = majority_vote([o for o in opinions if o is not None])
        return label, share, "majority"

    bert_label = opinion_label(bert_opinion)
    agent_labels = [opinion_label(item) for item in agent_opinions]
    agent_labels = [label for label in agent_labels if label is not None]
    unanimous = bool(agent_labels) and len(set(agent_labels)) == 1

    if mode == "bert_gate":
        high_conf = float(bert_opinion.confidence_level) >= bert_gate_confidence
        if high_conf and bert_label in {"yes", "no"}:
            # Only allow a unanimous panel override when it is a hard yes↔no flip.
            # Unanimous "maybe" must NOT veto a confident BioLinkBERT yes/no
            # (this previously hurt accuracy on balanced90).
            if (
                unanimous
                and agent_labels[0] in {"yes", "no"}
                and agent_labels[0] != bert_label
            ):
                label, share = majority_vote(agent_opinions)
                return label, share, "panel_unanimous_override"
            share = {label: 0.0 for label in _LABELS}
            if bert_label:
                share[bert_label] = 1.0
            return bert_label, share, "bert_gate"

        # Uncertain / maybe / lower confidence:
        # if the panel collapses to maybe-only, keep BioLinkBERT rather than
        # diluting a stronger classifier signal.
        if agent_labels and set(agent_labels) == {"maybe"} and bert_label in {"yes", "no"}:
            share = {label: 0.0 for label in _LABELS}
            share[bert_label] = 1.0
            return bert_label, share, "bert_keep_vs_panel_maybe"

        weights = [1.0] * len(agent_opinions) + [bert_vote_weight]
        label, share, _ = weighted_vote([*agent_opinions, bert_opinion], weights=weights)
        return label, share, "bert_weighted_uncertain"

    if mode == "bert_weighted":
        weights = [1.0] * len(agent_opinions) + [bert_vote_weight]
        label, share, _ = weighted_vote([*agent_opinions, bert_opinion], weights=weights)
        return label, share, "bert_weighted"

    label, share = majority_vote([*agent_opinions, bert_opinion])
    return label, share, "majority"


def final_labels_by_agent(entries: list[AgentRoundOpinion]) -> dict[str, str | None]:
    return {entry.agent_id: opinion_label(entry.opinion) for entry in entries}


async def aggregate_with_llm_director(
    patient_case: str,
    debate_history: list[list[AgentRoundOpinion]],
    biolinkbert_hint: str,
    *,
    supervisor: SupervisorAgent,
) -> str:
    """
    Aggregate a full debate by asking the LLM Director via `SupervisorAgent`.

    Returns the Director's `final_label` ("yes" | "no" | "maybe").
    """
    transcript_obj: dict[str, Any] = {
        "patient_case": patient_case,
        "rounds": [
            [
                {
                    "agent_id": entry.agent_id,
                    "persona": entry.persona,
                    "round": entry.round,
                    "opinion": entry.opinion.model_dump(),
                }
                for entry in round_entries
            ]
            for round_entries in debate_history
        ],
    }
    debate_transcript = json.dumps(transcript_obj, ensure_ascii=False)

    director_output = await supervisor.synthesize_decision(
        patient_case=patient_case,
        debate_transcript=debate_transcript,
        biolinkbert_hint=biolinkbert_hint,
    )
    setattr(supervisor, "last_director_output", director_output)
    return director_output.final_label


def build_consensus_decision(
    agent_opinions: list[ClinicalOpinion],
    *,
    predicted_label: str | None,
    vote_share: dict[str, float],
    safety_blocked: bool = False,
) -> ConsensusDecision:
    """
    Map panel aggregation into consensus / differential / escalation modes.

    Rules (PubMedQA-oriented):
    - High agreement + conclusive evidence -> consensus
    - Two strong competing labels -> differential (ranked hypotheses)
    - Safety block or unresolved low-confidence split -> escalation
    """
    labels = [opinion_label(opinion) for opinion in agent_opinions]
    labels = [label for label in labels if label is not None]
    if safety_blocked:
        return ConsensusDecision(
            mode="escalation",
            final_label=None,
            required_next_steps=_collect_next_steps(agent_opinions),
            grounding_score=_grounding_score(agent_opinions),
            safety_blocked=True,
            rationale="Safety audit blocked a definitive consensus label.",
        )

    if not labels or predicted_label is None:
        return ConsensusDecision(
            mode="escalation",
            final_label=None,
            required_next_steps=_collect_next_steps(agent_opinions),
            grounding_score=_grounding_score(agent_opinions),
            rationale="Panel did not produce a usable consensus label.",
        )

    share = max(vote_share.values()) if vote_share else 0.0
    unique = set(labels)
    conclusive = sum(
        1
        for opinion in agent_opinions
        if (opinion.evidence_conclusiveness or "").strip().lower() == "conclusive"
    )
    grounding = _grounding_score(agent_opinions)

    if len(unique) == 1 and share >= 0.75 and conclusive >= max(1, len(agent_opinions) // 2):
        return ConsensusDecision(
            mode="consensus",
            final_label=predicted_label,  # type: ignore[arg-type]
            ranked_hypotheses=[RankedHypothesis(label=predicted_label, score=share)],
            grounding_score=grounding,
            rationale="High panel agreement with conclusive evidence grounding.",
        )

    if len(unique) >= 2:
        ranked = [
            RankedHypothesis(label=label, score=float(vote_share.get(label, 0.0)))
            for label in sorted(unique, key=lambda item: vote_share.get(item, 0.0), reverse=True)
        ]
        top_share = ranked[0].score if ranked else 0.0
        second_share = ranked[1].score if len(ranked) > 1 else 0.0
        if top_share >= 0.34 and second_share >= 0.25:
            return ConsensusDecision(
                mode="differential",
                final_label=predicted_label,  # type: ignore[arg-type]
                ranked_hypotheses=ranked[:3],
                required_next_steps=_collect_next_steps(agent_opinions),
                grounding_score=grounding,
                rationale="Competing hypotheses remain after debate; returning ranked differential.",
            )

    if share < 0.5 or grounding < 0.35:
        return ConsensusDecision(
            mode="escalation",
            final_label="maybe" if predicted_label is None else predicted_label,  # type: ignore[arg-type]
            ranked_hypotheses=[
                RankedHypothesis(label=label, score=float(vote_share.get(label, 0.0)))
                for label in _LABELS
                if vote_share.get(label, 0.0) > 0
            ],
            required_next_steps=_collect_next_steps(agent_opinions),
            grounding_score=grounding,
            rationale="Insufficient agreement or grounding; escalate with next steps.",
        )

    return ConsensusDecision(
        mode="consensus",
        final_label=predicted_label,  # type: ignore[arg-type]
        ranked_hypotheses=[RankedHypothesis(label=predicted_label, score=share)],
        grounding_score=grounding,
        rationale="Panel reached a workable majority consensus.",
    )


def _grounding_score(agent_opinions: list[ClinicalOpinion]) -> float:
    if not agent_opinions:
        return 0.0
    grounded = 0
    for opinion in agent_opinions:
        if opinion.sources_used:
            grounded += 1
        elif (opinion.evidence_conclusiveness or "").strip().lower() == "conclusive":
            grounded += 1
    return grounded / len(agent_opinions)


def _collect_next_steps(agent_opinions: list[ClinicalOpinion]) -> list[str]:
    steps: list[str] = []
    for opinion in agent_opinions:
        steps.extend(opinion.required_further_tests)
        if opinion.missing_information.strip():
            steps.append(opinion.missing_information.strip())
    # Preserve order, drop duplicates.
    seen: set[str] = set()
    unique: list[str] = []
    for step in steps:
        key = step.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(step)
    return unique[:8]
