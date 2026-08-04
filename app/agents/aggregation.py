"""Aggregate multi-agent ClinicalOpinion outputs into a PubMedQA label (no supervisor)."""

from __future__ import annotations

from collections import Counter
import json
import re
from typing import Any

from app.agents.models import AgentRoundOpinion, ClinicalOpinion
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
