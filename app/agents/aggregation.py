"""Aggregate multi-agent ClinicalOpinion outputs into a PubMedQA label (no supervisor)."""

from __future__ import annotations

from collections import Counter
import json
import re
from typing import Any, Literal

from app.agents.models import (
    AgentRoundOpinion,
    ClinicalOpinion,
    ConsensusDecision,
    RankedHypothesis,
    SharedDebateReport,
    SupervisorDirectorOutput,
)
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


PanelMaybeVeto = Literal["unanimous", "majority", "off"]


def _panel_maybe_share(agent_labels: list[str]) -> float:
    if not agent_labels:
        return 0.0
    return sum(1 for label in agent_labels if label == "maybe") / len(agent_labels)


def _panel_maybe_veto_triggered(
    agent_labels: list[str],
    *,
    panel_maybe_veto: PanelMaybeVeto,
) -> bool:
    """Return True when the panel's maybe vote should override BioLinkBERT."""
    mode = (panel_maybe_veto or "unanimous").strip().lower()
    if mode == "off" or not agent_labels:
        return False
    maybe_share = _panel_maybe_share(agent_labels)
    if mode == "unanimous":
        return maybe_share >= 1.0
    if mode == "majority":
        return maybe_share > 0.5
    raise ValueError(f"Unsupported panel_maybe_veto: {panel_maybe_veto}")


def aggregate_pubmedqa_decision(
    agent_opinions: list[ClinicalOpinion],
    *,
    bert_opinion: ClinicalOpinion | None = None,
    mode: str = "majority",
    bert_gate_confidence: float = 0.90,
    bert_vote_weight: float = 3.0,
    panel_maybe_veto: PanelMaybeVeto = "unanimous",
) -> tuple[str | None, dict[str, float], str]:
    """
    Aggregate panel (+ optional BioLinkBERT) into a final yes/no/maybe.

    Modes:
      - majority: equal votes (confidence-weighted)
      - bert_weighted: BioLinkBERT vote amplified by bert_vote_weight
      - bert_gate: trust high-confidence BioLinkBERT yes/no unless the panel
        triggers a maybe veto (``panel_maybe_veto``) or unanimously hard-flips
        yes↔no; for maybe/low-confidence BERT, use weighted debate

    ``panel_maybe_veto`` (bert_gate only):
      - unanimous (default): all valid panel votes are maybe → final maybe
      - majority: maybe share > 0.5 → final maybe
      - off: legacy behavior (panel maybe does not override high-conf BERT)
    """
    mode = (mode or "majority").strip().lower()
    veto_mode: PanelMaybeVeto
    raw_veto = (panel_maybe_veto or "unanimous").strip().lower()
    if raw_veto not in {"unanimous", "majority", "off"}:
        raise ValueError(f"Unsupported panel_maybe_veto: {panel_maybe_veto}")
    veto_mode = raw_veto  # type: ignore[assignment]

    if bert_opinion is None or mode == "majority":
        opinions = [*agent_opinions, bert_opinion] if bert_opinion is not None else agent_opinions
        label, share = majority_vote([o for o in opinions if o is not None])
        return label, share, "majority"

    bert_label = opinion_label(bert_opinion)
    agent_labels = [opinion_label(item) for item in agent_opinions]
    agent_labels = [label for label in agent_labels if label is not None]
    unanimous = bool(agent_labels) and len(set(agent_labels)) == 1

    if mode == "bert_gate":
        # Panel maybe is clinically absolute when enabled: overrides BERT on any path.
        if _panel_maybe_veto_triggered(agent_labels, panel_maybe_veto=veto_mode):
            share = {label: 0.0 for label in _LABELS}
            share["maybe"] = 1.0
            return "maybe", share, "panel_maybe_veto"

        high_conf = float(bert_opinion.confidence_level) >= bert_gate_confidence
        if high_conf and bert_label in {"yes", "no"}:
            # Unanimous hard yes↔no flip still overrides confident BioLinkBERT.
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

        # Uncertain / maybe / lower-confidence BERT → weighted panel+BERT vote.
        # Legacy bert_keep_vs_panel_maybe is retired when veto != off (handled above);
        # with veto=off, preserve old "keep BERT vs panel-maybe-only" behavior.
        if (
            veto_mode == "off"
            and agent_labels
            and set(agent_labels) == {"maybe"}
            and bert_label in {"yes", "no"}
        ):
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


def is_fallback_opinion(opinion: ClinicalOpinion) -> bool:
    sources = [str(s).strip().lower() for s in (opinion.sources_used or [])]
    if "fallback" in sources:
        return True
    missing = (opinion.missing_information or "").lower()
    return "invalid" in missing and "json" in missing


def opinion_validity_weight(opinion: ClinicalOpinion) -> float:
    """MedARC-style validity weight: zero-out system-fallback opinions."""
    if is_fallback_opinion(opinion):
        return 0.0
    try:
        conf = float(opinion.confidence_level)
    except (TypeError, ValueError):
        conf = 0.4
    return min(1.0, max(0.05, conf))


def confidence_aware_vote(
    opinions: list[ClinicalOpinion],
) -> tuple[str | None, dict[str, float], dict[str, float]]:
    """Confidence × validity weighted vote; fallbacks contribute 0."""
    weights = [opinion_validity_weight(op) for op in opinions]
    return weighted_vote(opinions, weights=weights)


def _format_opinion_for_transcript(entry: AgentRoundOpinion) -> str:
    """Full per-agent opinion block (all pros/cons) for Director contamination."""
    opinion = entry.opinion
    conf = float(opinion.confidence_level)
    conclusiveness = (opinion.evidence_conclusiveness or "").strip() or "unspecified"
    lines = [
        f"[{entry.agent_id}|{entry.persona}] "
        f"(conf={conf:.2f}, {conclusiveness}): {opinion.top_1_diagnosis}"
    ]
    for pro in opinion.pros or []:
        text = str(pro).strip()
        if text:
            lines.append(f"  Pro: {text}")
    for con in opinion.cons or []:
        text = str(con).strip()
        if text:
            lines.append(f"  Con: {text}")
    for flag in opinion.red_flags or []:
        text = str(flag).strip()
        if text:
            lines.append(f"  RedFlag: {text}")
    missing = str(opinion.missing_information or "").strip()
    if missing:
        lines.append(f"  Missing: {missing}")
    return "\n".join(lines)


def build_full_debate_transcript(
    debate_history: list[list[AgentRoundOpinion]],
    *,
    shared_report: SharedDebateReport | None = None,
) -> str:
    """Multi-round transcript with unresolved conflicts kept visible for the Director.

    Unlike ``build_debate_brief``, this keeps every round and every pro/con so the
    Director sees the full contaminated discussion, not a sanitized final-round summary.
    """
    parts: list[str] = []
    for round_idx, round_entries in enumerate(debate_history, start=1):
        parts.append(f"=== ROUND {round_idx} ===")
        if not round_entries:
            parts.append("(no opinions)")
            continue
        for entry in round_entries:
            parts.append(_format_opinion_for_transcript(entry))
        labels = [
            lab
            for lab in (opinion_label(e.opinion) for e in round_entries)
            if lab is not None
        ]
        if labels:
            counts = Counter(labels)
            tally = ", ".join(f"{lab}={n}" for lab, n in sorted(counts.items()))
            parts.append(f"Round {round_idx} label tally: {tally}")
            if len(counts) > 1:
                split = "; ".join(
                    f"{e.agent_id}={opinion_label(e.opinion)}" for e in round_entries
                )
                parts.append(f"Round {round_idx} CONFLICT: {split}")

    if debate_history:
        final = debate_history[-1]
        final_labels = {
            e.agent_id: opinion_label(e.opinion)
            for e in final
            if opinion_label(e.opinion) is not None
        }
        unique = {lab for lab in final_labels.values() if lab}
        if len(unique) > 1:
            parts.append("=== FINAL PANEL CONFLICT (unresolved) ===")
            parts.append(
                "; ".join(f"{aid}={lab}" for aid, lab in sorted(final_labels.items()))
            )

    if shared_report is not None:
        parts.append("=== SUPERVISOR SHARED REPORT (conflict-preserving) ===")
        from app.agents.prompts import format_moderation_nl

        parts.append(format_moderation_nl(shared_report))

    return "\n".join(parts).strip()


def build_debate_brief(
    debate_history: list[list[AgentRoundOpinion]],
    *,
    shared_report: SharedDebateReport | None = None,
) -> dict[str, Any]:
    """Compact MedARC-style brief (telemetry / SFT meta; not the Director primary input)."""
    final = debate_history[-1] if debate_history else []
    panel = []
    for entry in final:
        label = opinion_label(entry.opinion)
        weight = opinion_validity_weight(entry.opinion)
        panel.append(
            {
                "agent_id": entry.agent_id,
                "persona": entry.persona,
                "role": (
                    "author_conclusion_reader"
                    if entry.agent_id == "generalist"
                    else (
                        "uncertainty_advocate"
                        if entry.agent_id == "uncertainty_advocate"
                        else entry.persona
                    )
                ),
                "label": label,
                "confidence": float(entry.opinion.confidence_level),
                "validity_weight": weight,
                "is_fallback": is_fallback_opinion(entry.opinion),
                "evidence_conclusiveness": entry.opinion.evidence_conclusiveness,
                "pros": (entry.opinion.pros or [])[:1],
                "cons": (entry.opinion.cons or [])[:1],
            }
        )
    conf_label, conf_share, conf_scores = confidence_aware_vote(
        [entry.opinion for entry in final]
    )
    advocate = next((p for p in panel if p["agent_id"] == "uncertainty_advocate"), None)
    generalist = next((p for p in panel if p["agent_id"] == "generalist"), None)
    return {
        "rounds_completed": len(debate_history),
        "confidence_aware_panel_vote": conf_label,
        "confidence_aware_share": conf_share,
        "confidence_aware_scores": conf_scores,
        "dual_read": {
            "author_conclusion_reader": generalist,
            "uncertainty_advocate": advocate,
        },
        "panel": panel,
        "shared_report": shared_report.model_dump() if shared_report is not None else None,
    }


def _parse_biolinkbert_label(biolinkbert_hint: str | None) -> str | None:
    """Extract yes/no/maybe from director hint JSON or raw text."""
    raw = (biolinkbert_hint or "").strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return extract_label(str(data.get("label") or ""))
    except Exception:
        pass
    return extract_label(raw)


def apply_maybe_director_gate(
    output: SupervisorDirectorOutput,
    *,
    patient_case: str,
    final_opinions: list[AgentRoundOpinion],
    shared_report: SharedDebateReport | None = None,
    advocate_maybe_confidence: float = 0.75,
    bert_label: str | None = None,
    rounds_completed: int = 1,
) -> SupervisorDirectorOutput:
    """
    Selective maybe gate on top of LLM director output.

    Goals:
    - Recover PubMedQA ``maybe`` when question coverage is none / strong uncertainty.
    - Avoid false maybe (v4): do not downgrade on ``partial`` alone when the panel
      confidence-aware vote and BioLinkBERT already agree with the director yes/no.
    """
    from app.agents.heuristics import abstract_suggests_inconclusive

    if output.final_label == "maybe":
        return output

    usable = [e for e in final_opinions if not is_fallback_opinion(e.opinion)]
    panel_labels = [opinion_label(e.opinion) for e in usable]
    panel_labels = [lab for lab in panel_labels if lab is not None]
    panel_unanimous_binary = (
        bool(panel_labels)
        and len(set(panel_labels)) == 1
        and panel_labels[0] in {"yes", "no"}
    )
    panel_agrees_with_director = panel_unanimous_binary and panel_labels[0] == output.final_label
    mean_conf = (
        sum(float(e.opinion.confidence_level) for e in usable) / len(usable) if usable else 0.0
    )
    bert_norm = (bert_label or "").strip().lower() or None
    if bert_norm not in {"yes", "no", "maybe"}:
        bert_norm = None

    # Hard protect: unanimous high-conf panel matching director yes/no (+ BERT if present).
    bert_ok = bert_norm is None or bert_norm == output.final_label
    if panel_agrees_with_director and mean_conf >= 0.8 and bert_ok:
        return output

    coverage = (getattr(output, "question_coverage", None) or "full").strip().lower()
    checklist_hits = sum(
        [
            not bool(output.primary_endpoint_answers_question),
            not bool(output.findings_decisive_for_question),
            bool(output.authors_state_uncertainty),
        ]
    )

    reasons: list[str] = []

    # Path A: abstract does not fully cover the research question.
    if coverage == "none":
        reasons.append("question_coverage=none")
    elif coverage == "partial" and (
        output.consensus_type in {"differential", "escalation"}
        or not output.findings_decisive_for_question
        or checklist_hits >= 1
    ):
        reasons.append("question_coverage=partial + uncertainty signal")

    # Path B: director already marked differential and findings not decisive.
    if (
        output.consensus_type in {"differential", "escalation"}
        and not output.findings_decisive_for_question
    ):
        reasons.append("differential + non-decisive findings")

    # Path C: strong dual checklist.
    if checklist_hits >= 2:
        reasons.append(f"checklist_hits={checklist_hits}")

    # Path D: uncertainty expert maybe + residual + specific abstract cue (tight).
    from app.agents.prompts import PUBMEDQA_UNCERTAINTY_PERSONAS

    uncertainty_experts = [
        e
        for e in usable
        if e.agent_id in PUBMEDQA_UNCERTAINTY_PERSONAS
        or e.persona in PUBMEDQA_UNCERTAINTY_PERSONAS
    ]
    for expert in uncertainty_experts:
        expert_label = opinion_label(expert.opinion)
        if (
            expert_label == "maybe"
            and float(expert.opinion.confidence_level) >= advocate_maybe_confidence
        ):
            residual = list(shared_report.residual_uncertainty) if shared_report else []
            if residual and abstract_suggests_inconclusive(patient_case):
                role = expert.agent_id or expert.persona or "uncertainty_expert"
                reasons.append(f"{role} maybe + residual + abstract cue")
                break

    # Path E: after 3+ rounds, uncertainty_advocate holding high-confidence
    # maybe has survived multiple debate challenges — treat as a validated gap.
    if rounds_completed >= 3:
        for expert in uncertainty_experts:
            role = (expert.agent_id or expert.persona or "").strip().lower()
            expert_label = opinion_label(expert.opinion)
            if (
                expert_label == "maybe"
                and float(expert.opinion.confidence_level) >= advocate_maybe_confidence
            ):
                reasons.append(
                    f"{role} persistent maybe after {rounds_completed} rounds"
                )

    if not reasons:
        return output

    conf_label, conf_share, _ = confidence_aware_vote(
        [e.opinion for e in usable] or [e.opinion for e in final_opinions]
    )
    panel_bert_agree_binary = (
        output.final_label in {"yes", "no"}
        and conf_label == output.final_label
        and float(conf_share.get(output.final_label, 0.0)) >= 0.5
        and (bert_norm is None or bert_norm == output.final_label)
    )

    # v5 safeguard: if panel vote + BERT already back the director binary, only allow
    # the strongest maybe paths (coverage=none / auditor+abstract cue).
    # Blocks false maybe from partial / soft differential / flaky checklist alone.
    if panel_bert_agree_binary:
        reasons = [
            r
            for r in reasons
            if r.startswith("question_coverage=none")
            or "abstract cue" in r
            or "persistent maybe" in r
        ]
        if not reasons:
            return output

    return SupervisorDirectorOutput(
        final_label="maybe",
        consensus_type="differential"
        if output.consensus_type == "consensus"
        else output.consensus_type,
        rationale=(output.rationale + " | maybe_gate: " + "; ".join(reasons)).strip(),
        debate_conflict_level=output.debate_conflict_level,
        conclusiveness_score=output.conclusiveness_score,
        unresolved_contradictions=list(output.unresolved_contradictions),
        primary_endpoint_answers_question=output.primary_endpoint_answers_question,
        findings_decisive_for_question=output.findings_decisive_for_question,
        authors_state_uncertainty=output.authors_state_uncertainty,
        question_coverage=coverage if coverage in {"full", "partial", "none"} else "full",
    )


async def aggregate_with_llm_director(
    patient_case: str,
    debate_history: list[list[AgentRoundOpinion]],
    biolinkbert_hint: str,
    *,
    supervisor: SupervisorAgent,
    shared_report: SharedDebateReport | None = None,
    director_maybe_gate: Literal["off", "legacy"] = "off",
) -> str:
    """
    Aggregate a full debate by asking the LLM Director via `SupervisorAgent`.

    Passes the **full multi-round conflict transcript** (every agent turn + shared
    report contradictions). Compact ``build_debate_brief`` is kept only as unused
    telemetry metadata on the supervisor for debugging.

    By default (``director_maybe_gate="off"``) the Director's ``final_label`` and
    ``rationale`` are taken as-is — Python does not rewrite the verdict. Pass
    ``director_maybe_gate="legacy"`` to re-enable the post-hoc
    ``apply_maybe_director_gate`` override (ablation only).

    Returns the Director's `final_label` ("yes" | "no" | "maybe").
    """
    gate_mode = (director_maybe_gate or "off").strip().lower()
    if gate_mode not in {"off", "legacy"}:
        raise ValueError(f"Unsupported director_maybe_gate: {director_maybe_gate}")

    # Prefer latest moderator shared report if caller did not pass one.
    if shared_report is None:
        last_mod = getattr(supervisor, "last_moderation_output", None)
        if last_mod is not None and hasattr(last_mod, "as_shared_report"):
            shared_report = last_mod.as_shared_report()

    transcript = build_full_debate_transcript(
        debate_history, shared_report=shared_report
    )
    brief = build_debate_brief(debate_history, shared_report=shared_report)
    shared_report_text = json.dumps(
        shared_report.model_dump() if shared_report is not None else {},
        ensure_ascii=False,
    )
    # Telemetry only — never used to override final_label.
    setattr(supervisor, "last_debate_brief", brief)
    setattr(supervisor, "last_debate_transcript", transcript)

    director_output = await supervisor.synthesize_decision(
        patient_case=patient_case,
        debate_transcript=transcript,
        biolinkbert_hint=biolinkbert_hint,
        shared_report=shared_report_text,
        debate_brief=None,
    )
    if gate_mode == "legacy":
        final_opinions = debate_history[-1] if debate_history else []
        director_output = apply_maybe_director_gate(
            director_output,
            patient_case=patient_case,
            final_opinions=final_opinions,
            shared_report=shared_report,
            bert_label=_parse_biolinkbert_label(biolinkbert_hint),
            rounds_completed=len(debate_history),
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
