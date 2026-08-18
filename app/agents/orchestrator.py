"""Debate orchestrator with active supervisor moderation (MAC / MedAgent style rounds)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import json
from typing import Any, Literal

from app.agents.agent import ClinicalAgent
from app.agents.aggregation import opinion_label
from app.agents.heuristics import (
    INCONCLUSIVE_ABSTRACT_PHRASES,
    abstract_suggests_inconclusive,
)
from app.agents.models import (
    AgentRoundOpinion,
    ClinicalOpinion,
    DebateResult,
    SupervisorModerationOutput,
)
from app.agents.supervisor_agent import SupervisorAgent
from app.agents.uncertainty import _label_entropy
from app.agents.prompts import (
    PeerContextMode,
    format_moderation_nl,
)

DEFAULT_PERSONAS: tuple[tuple[str, str], ...] = (
    ("generalist", "generalist"),
    ("evidence_skeptic", "evidence_skeptic"),
    ("differential_expander", "differential_expander"),
    ("safety_officer", "safety_officer"),
)

PUBMEDQA_PERSONAS: tuple[tuple[str, str], ...] = (
    ("generalist", "generalist"),
    ("evidence_skeptic", "evidence_skeptic"),
    ("differential_expander", "differential_expander"),
    ("uncertainty_advocate", "uncertainty_advocate"),
)

DebateMode = Literal["moderated", "peer", "hybrid"]
BlindCriticMode = Literal["all-rounds", "r1-only", "off"]
SafetyRedFlagMode = Literal["halt", "escalate-label", "defer"]
SupervisorFailMode = Literal["peer-round", "peer-rest", "empty-defer"]
EarlyExitFn = Callable[[int, list[AgentRoundOpinion], str], bool]

# Personas that can be kept blind to BioLinkBERT (see ``blind_critic``).
_BLIND_HINT_PERSONAS: frozenset[str] = frozenset({"uncertainty_advocate"})
# Backward-compatible alias.
_BLIND_HINT_PERSONAS_R1 = _BLIND_HINT_PERSONAS

_ARCHITECTURE_BY_MODE: dict[DebateMode, str] = {
    "moderated": "supervised_moderation",
    "peer": "peer_round_robin",
    "hybrid": "hybrid_supervised_peer",
}


class DebateOrchestrator:
    """
    Runs multi-round peer debate among ClinicalAgent instances.

    Round 1 ("independent opinion"): every agent answers in parallel with no
    peer context, so nobody anchors on somebody else's first take.

    Blind critic (``blind_critic``): by default ``uncertainty_advocate`` never
    sees BioLinkBERT hints (``all-rounds``), to avoid authority bias after an
    independent round-1 ``maybe``. Legacy ``r1-only`` restores hint from round 2;
    ``off`` exposes the hint from round 1.

    Round 2+ depends on ``debate_mode``:

    - ``moderated`` (default): supervisor moderates, then all agents revise
      concurrently using supervisor instructions only (legacy behavior).
    - ``peer``: round-robin peer debate; each agent sees prior-round opinions
      plus peers who already spoke this round. No supervisor moderation.
    - ``hybrid``: supervisor moderates, then round-robin peer debate where each
      agent also sees supervisor agreements/contradictions/instructions plus
      peer opinions from the previous round.

    With ``adaptive_rounds=True``, the debate runs at least ``min_rounds`` and
    continues up to ``max_rounds`` while conflict remains unresolved
    (label entropy / supervisor contradictions / non-unanimous panel).

    Optional early-exit skips later rounds when the panel already agrees
    (typically via ``check_early_exit_asymmetric_veto``: binary unanimity plus
    uncertainty_advocate veto on ``maybe`` / low confidence).

    Safety red flags (``safety_red_flag``): when ``safety_officer`` sets
    ``safety_passed=False`` and ``immediate_intervention_required=True``:
    - ``halt`` (default): stop the debate immediately (no further rounds)
    - ``escalate-label``: same halt; eval may force ``maybe`` + escalation
    - ``defer``: legacy — inject ``RED FLAG DETECTED`` into the next round only

    Supervisor parse failure (``supervisor_fail``): after retry at temperature 0,
    - ``peer-round`` (default): run this round as peer round-robin (no empty moderation)
    - ``peer-rest``: same, then keep peer for remaining rounds of the case
    - ``empty-defer``: legacy empty moderation fallback injected into moderated/hybrid
    """

    ARCHITECTURE: str = "supervised_moderation"

    def __init__(
        self,
        agents: list[ClinicalAgent],
        *,
        rounds: int = 3,
        min_rounds: int | None = None,
        max_rounds: int | None = None,
        adaptive_rounds: bool = False,
        conflict_entropy_threshold: float = 0.35,
        debate_mode: DebateMode = "moderated",
        blind_critic: BlindCriticMode = "all-rounds",
        safety_red_flag: SafetyRedFlagMode = "halt",
        peer_context: PeerContextMode = "nl",
        supervisor_fail: SupervisorFailMode = "peer-round",
        early_exit: EarlyExitFn | None = None,
        agent_concurrency: int = 4,
        supervisor_backend: Any | None = None,
    ) -> None:
        if len(agents) < 2:
            raise ValueError("DebateOrchestrator requires at least 2 agents.")
        if agent_concurrency < 1:
            raise ValueError("agent_concurrency must be >= 1.")
        if debate_mode not in _ARCHITECTURE_BY_MODE:
            raise ValueError(f"Unsupported debate_mode: {debate_mode}")
        blind_mode = (blind_critic or "all-rounds").strip().lower()
        if blind_mode not in {"all-rounds", "r1-only", "off"}:
            raise ValueError(f"Unsupported blind_critic: {blind_critic}")
        safety_mode = (safety_red_flag or "halt").strip().lower()
        if safety_mode not in {"halt", "escalate-label", "defer"}:
            raise ValueError(f"Unsupported safety_red_flag: {safety_red_flag}")
        peer_mode = (peer_context or "nl").strip().lower()
        if peer_mode not in {"nl", "compact-json", "full-json"}:
            raise ValueError(f"Unsupported peer_context: {peer_context}")
        fail_mode = (supervisor_fail or "peer-round").strip().lower()
        if fail_mode not in {"peer-round", "peer-rest", "empty-defer"}:
            raise ValueError(f"Unsupported supervisor_fail: {supervisor_fail}")

        resolved_max = max_rounds if max_rounds is not None else rounds
        resolved_min = min_rounds if min_rounds is not None else (2 if adaptive_rounds else resolved_max)
        if resolved_min < 2 or resolved_max > 5:
            raise ValueError("rounds must be between 2 and 5 (min_rounds/max_rounds).")
        if resolved_min > resolved_max:
            raise ValueError("min_rounds cannot exceed max_rounds.")
        if not adaptive_rounds and (resolved_min != resolved_max):
            # Fixed-length mode: use max as the exact round count.
            resolved_min = resolved_max

        self.agents = agents
        self.rounds = resolved_max
        self.min_rounds = resolved_min
        self.max_rounds = resolved_max
        self.adaptive_rounds = adaptive_rounds
        self.conflict_entropy_threshold = conflict_entropy_threshold
        self.debate_mode = debate_mode
        self.blind_critic: BlindCriticMode = blind_mode  # type: ignore[assignment]
        self.safety_red_flag: SafetyRedFlagMode = safety_mode  # type: ignore[assignment]
        self.peer_context: PeerContextMode = peer_mode  # type: ignore[assignment]
        self.supervisor_fail: SupervisorFailMode = fail_mode  # type: ignore[assignment]
        self.early_exit = early_exit
        self.agent_concurrency = agent_concurrency
        self.early_exits = 0
        self.adaptive_stops = 0
        self.safety_halts = 0
        self.supervisor_failovers = 0
        self.ARCHITECTURE = _ARCHITECTURE_BY_MODE[debate_mode]
        for agent in self.agents:
            setattr(agent, "peer_context", self.peer_context)
        self.supervisor = SupervisorAgent(
            backend=supervisor_backend if supervisor_backend is not None else agents[0].backend,
            peer_context=self.peer_context,
        )

    async def run(self, patient_case: str) -> DebateResult:
        history: list[list[AgentRoundOpinion]] = []
        supervisor_moderation: list[SupervisorModerationOutput] = []
        pending_red_flag_instruction: str | None = None
        safety_halted = False
        safety_red_flag_reason: str | None = None
        force_peer_rest = False

        for round_number in range(1, self.max_rounds + 1):
            if round_number == 1:
                round_opinions = await self._run_independent_round(patient_case)
            else:
                previous_round = history[-1]
                supervisor_prefix: list[AgentRoundOpinion] = []
                failover_to_peer = False
                skip_supervisor = force_peer_rest or self.debate_mode == "peer"

                if not skip_supervisor and self.debate_mode in ("moderated", "hybrid"):
                    moderation = await self.supervisor.moderate_round(
                        patient_case=patient_case,
                        agents_opinions=previous_round,
                    )
                    if (
                        self.supervisor.last_moderation_failed
                        and self.supervisor_fail != "empty-defer"
                    ):
                        failover_to_peer = True
                        self.supervisor_failovers += 1
                        if self.supervisor_fail == "peer-rest":
                            force_peer_rest = True
                    else:
                        if pending_red_flag_instruction:
                            moderation.round_instructions.insert(
                                0, pending_red_flag_instruction
                            )
                        supervisor_moderation.append(moderation)
                        supervisor_prefix = _moderation_as_supervisor_context(
                            moderation_output=moderation,
                            round_number=round_number,
                            peer_context=self.peer_context,
                        )

                if failover_to_peer or force_peer_rest or self.debate_mode == "peer":
                    round_opinions = await self._run_round_robin_round(
                        patient_case,
                        round_number=round_number,
                        previous_round=previous_round,
                        context_prefix=None,
                    )
                elif self.debate_mode == "moderated":
                    round_opinions = await self._run_moderated_round(
                        patient_case=patient_case,
                        round_number=round_number,
                        supervisor_context=supervisor_prefix,
                    )
                else:
                    round_opinions = await self._run_round_robin_round(
                        patient_case,
                        round_number=round_number,
                        previous_round=previous_round,
                        context_prefix=supervisor_prefix,
                    )
            history.append(round_opinions)

            critical_flag = _pending_red_flag_instruction(round_opinions)
            if critical_flag:
                if self.safety_red_flag == "defer":
                    pending_red_flag_instruction = critical_flag
                else:
                    # halt / escalate-label: hard stop — do not continue debating.
                    safety_halted = True
                    safety_red_flag_reason = critical_flag
                    self.safety_halts += 1
                    break
            else:
                pending_red_flag_instruction = None

            if round_number >= self.max_rounds:
                break

            if (
                self.early_exit is not None
                and self.early_exit(round_number, round_opinions, patient_case)
            ):
                self.early_exits += 1
                break

            if self.adaptive_rounds and round_number >= self.min_rounds:
                latest_moderation = supervisor_moderation[-1] if supervisor_moderation else None
                if not should_continue_debate(
                    round_opinions,
                    moderation=latest_moderation,
                    entropy_threshold=self.conflict_entropy_threshold,
                ):
                    self.adaptive_stops += 1
                    break

        final_opinions = history[-1] if history else []
        exhausted = (
            not safety_halted
            and len(history) >= self.max_rounds
            and fundamental_panel_conflict(final_opinions)
        )
        return DebateResult(
            patient_case=patient_case,
            rounds=history,
            final_opinions=final_opinions,
            supervisor_moderation=supervisor_moderation,
            shared_report=(
                supervisor_moderation[-1].as_shared_report()
                if supervisor_moderation
                else None
            ),
            safety_halted=safety_halted,
            safety_red_flag_reason=safety_red_flag_reason,
            exhausted_without_consensus=exhausted,
        )

    async def _run_independent_round(self, patient_case: str) -> list[AgentRoundOpinion]:
        """Round 1: every agent answers concurrently with no peer context."""
        semaphore = asyncio.Semaphore(self.agent_concurrency)

        async def _one(agent: ClinicalAgent) -> AgentRoundOpinion:
            async with semaphore:
                return await self._speak(
                    agent,
                    patient_case,
                    round_number=1,
                    context=None,
                )

        return list(await asyncio.gather(*[_one(agent) for agent in self.agents]))

    async def _run_round_robin_round(
        self,
        patient_case: str,
        *,
        round_number: int,
        previous_round: list[AgentRoundOpinion],
        context_prefix: list[AgentRoundOpinion] | None = None,
    ) -> list[AgentRoundOpinion]:
        """Round 2+: sequential turns; each agent sees prefix + peer opinions."""
        spoken_so_far: list[AgentRoundOpinion] = []
        round_opinions: list[AgentRoundOpinion] = []
        prefix = list(context_prefix or [])

        for agent in self.agents:
            peer_context = _round_robin_context(
                agent.agent_id,
                previous_round,
                spoken_so_far,
            )
            context = prefix + peer_context
            entry = await self._speak(
                agent,
                patient_case,
                round_number=round_number,
                context=context or None,
            )
            spoken_so_far.append(entry)
            round_opinions.append(entry)
        return round_opinions

    async def _run_moderated_round(
        self,
        *,
        patient_case: str,
        round_number: int,
        supervisor_context: list[AgentRoundOpinion],
    ) -> list[AgentRoundOpinion]:
        """Round 2+ moderated mode: concurrent revision using supervisor instructions."""
        semaphore = asyncio.Semaphore(self.agent_concurrency)

        async def _one(agent: ClinicalAgent) -> AgentRoundOpinion:
            async with semaphore:
                return await self._speak(
                    agent,
                    patient_case,
                    round_number=round_number,
                    context=supervisor_context,
                )

        return list(await asyncio.gather(*[_one(agent) for agent in self.agents]))

    def _should_include_evidence_hint(self, agent: ClinicalAgent, round_number: int) -> bool:
        """Whether BioLinkBERT hint is shown to this agent this round."""
        if self.blind_critic == "off":
            return True
        is_blind_persona = (
            agent.persona in _BLIND_HINT_PERSONAS or agent.agent_id in _BLIND_HINT_PERSONAS
        )
        if not is_blind_persona:
            return True
        if self.blind_critic == "all-rounds":
            return False
        # r1-only: hide hint in round 1, restore from round 2.
        return round_number != 1

    async def _speak(
        self,
        agent: ClinicalAgent,
        patient_case: str,
        *,
        round_number: int,
        context: list[AgentRoundOpinion] | None,
        include_evidence_hint: bool | None = None,
    ) -> AgentRoundOpinion:
        if include_evidence_hint is None:
            include_evidence_hint = self._should_include_evidence_hint(agent, round_number)
        opinion = await agent.generate_opinion(
            patient_case,
            context=context,
            include_evidence_hint=include_evidence_hint,
        )
        return AgentRoundOpinion(
            agent_id=agent.agent_id,
            persona=agent.persona,
            round=round_number,
            opinion=opinion,
        )


def labels_unanimous(round_opinions: list[AgentRoundOpinion]) -> bool:
    labels = [opinion_label(entry.opinion) for entry in round_opinions]
    labels = [label for label in labels if label is not None]
    return bool(labels) and len(set(labels)) == 1


_CONVICTION_ROLES = frozenset({"generalist", "differential_expander"})
_GUARDIAN_ROLES = frozenset({"evidence_skeptic", "uncertainty_advocate"})

EXHAUSTED_NO_CONSENSUS_NOTE = (
    "Debate exhausted max rounds with a fundamental panel split "
    "(2-2 or skeptic+advocate maybe); defaulting to maybe."
)


def _entry_role(entry: AgentRoundOpinion) -> str:
    return (entry.agent_id or entry.persona or "").strip().lower()


def fundamental_panel_conflict(round_opinions: list[AgentRoundOpinion]) -> bool:
    """True only for a *fundamental* end-of-debate split.

    Activate when:
    - even 2-2 label split, or
    - both guardians (evidence_skeptic and uncertainty_advocate) vote maybe, or
    - conviction bloc (generalist + expander) shares a binary yes/no while both
      guardians dissent (no or maybe).

    A 3-1 where only the advocate votes maybe is *not* fundamental: the
    director may still assign yes/no.
    """
    labels_by_role: dict[str, str] = {}
    labels: list[str] = []
    for entry in round_opinions:
        lab = opinion_label(entry.opinion)
        if lab is None:
            continue
        labels.append(lab)
        labels_by_role[_entry_role(entry)] = lab
    if len(labels) < 2:
        return False

    skeptic = labels_by_role.get("evidence_skeptic")
    advocate = labels_by_role.get("uncertainty_advocate")
    if skeptic == "maybe" and advocate == "maybe":
        return True

    counts: dict[str, int] = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    ranked = sorted(counts.values(), reverse=True)
    if len(ranked) >= 2 and ranked[0] == 2 and ranked[1] == 2:
        return True

    conviction = [
        labels_by_role[role]
        for role in _CONVICTION_ROLES
        if role in labels_by_role
    ]
    guardians = [
        labels_by_role[role]
        for role in _GUARDIAN_ROLES
        if role in labels_by_role
    ]
    if len(conviction) == 2 and len(guardians) == 2:
        conviction_label = conviction[0]
        if (
            conviction_label in {"yes", "no"}
            and conviction[1] == conviction_label
            and all(label != conviction_label for label in guardians)
        ):
            return True
    return False


def apply_exhausted_no_consensus_override(
    predicted: str | None,
    *,
    exhausted_without_consensus: bool,
) -> tuple[str | None, bool]:
    """Force maybe after max rounds only on a fundamental panel split.

    Returns ``(label, overridden)``. 3-1 advocate-only maybe, early exits,
    adaptive stops, and safety halts are left to the director. Already-``maybe``
    labels are not rewritten.
    """
    if not exhausted_without_consensus or predicted == "maybe":
        return predicted, False
    return "maybe", True


DEFAULT_ADVOCATE_VETO_CONFIDENCE = 0.65


def _safe_confidence(value: Any, default: float = 1.0) -> float:
    """Parse confidence to float in [0, 1]; fall back to *default* on bad input."""
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return default
    if conf != conf:  # NaN
        return default
    return min(1.0, max(0.0, conf))


def check_early_exit_asymmetric_veto(
    round_opinions: list[AgentRoundOpinion],
    *,
    patient_case: str | None = None,
    advocate_confidence_threshold: float = DEFAULT_ADVOCATE_VETO_CONFIDENCE,
) -> bool:
    """
    Early-exit with asymmetric veto for the ``maybe`` class.

    Returns True only when the panel is unanimously ``yes`` or ``no`` *and*
    ``uncertainty_advocate`` does not veto. The advocate vetoes when they label
    ``maybe`` or their confidence is below ``advocate_confidence_threshold``.

    Also blocked when ``patient_case`` matches light inconclusiveness heuristics
    (e.g. ``further research is needed``), or when a majority of agents mark
    evidence as inconclusive.

    BioLinkBERT is intentionally ignored (option A).
    """
    if not round_opinions:
        return False

    if patient_case and abstract_suggests_inconclusive(patient_case):
        return False

    inconclusive_marks = sum(
        1
        for entry in round_opinions
        if (entry.opinion.evidence_conclusiveness or "").strip().lower() == "inconclusive"
    )
    # Require a strong majority inconclusive marks (avoid blocking on 2/4 noise).
    if inconclusive_marks >= max(3, len(round_opinions) - 1):
        return False

    advocate = next(
        (
            entry
            for entry in round_opinions
            if entry.agent_id == "uncertainty_advocate"
            or entry.persona == "uncertainty_advocate"
        ),
        None,
    )
    if advocate is not None:
        confidence = _safe_confidence(advocate.opinion.confidence_level, default=1.0)
        label = (opinion_label(advocate.opinion) or "").strip().lower()
        if label == "maybe" or confidence < advocate_confidence_threshold:
            return False

    labels = [
        (opinion_label(entry.opinion) or "").strip().lower()
        for entry in round_opinions
    ]
    labels = [label for label in labels if label]
    if len(labels) != len(round_opinions):
        return False
    unique = set(labels)
    if len(unique) != 1:
        return False
    return next(iter(unique)) in ("yes", "no")


def should_continue_debate(
    round_opinions: list[AgentRoundOpinion],
    *,
    moderation: SupervisorModerationOutput | None = None,
    entropy_threshold: float = 0.35,
) -> bool:
    """
    Return True when the debate should escalate to another round.

    Continue when:
    - panel is not unanimous, or
    - normalized label entropy exceeds threshold, or
    - supervisor reported unresolved contradictions / residual uncertainty, or
    - shared report author_conclusion is maybe/unclear while panel is binary.
    """
    if moderation is not None and moderation.contradictions:
        return True
    if moderation is not None and moderation.residual_uncertainty and len(moderation.residual_uncertainty) >= 2:
        return True
    labels = [opinion_label(entry.opinion) for entry in round_opinions]
    labels = [label for label in labels if label is not None]
    if not labels:
        return True
    if len(set(labels)) > 1:
        return True
    return _label_entropy(labels) > entropy_threshold


def _round_robin_context(
    agent_id: str,
    previous_round: list[AgentRoundOpinion],
    spoken_so_far: list[AgentRoundOpinion],
) -> list[AgentRoundOpinion]:
    """Previous round's peers (excluding self) followed by this round's turns so far."""
    previous_peers = [entry for entry in previous_round if entry.agent_id != agent_id]
    current_peers = [entry for entry in spoken_so_far if entry.agent_id != agent_id]
    return previous_peers + current_peers


def _pending_red_flag_instruction(round_opinions: list[AgentRoundOpinion]) -> str | None:
    """
    If safety_officer reports a critical safety situation, return a high-priority
    instruction to prepend to the next supervisor moderation output.
    """
    for entry in round_opinions:
        if entry.agent_id != "safety_officer":
            continue
        safety = getattr(entry.opinion, "safety_opinion", None)
        if safety is None:
            continue
        if not safety.safety_passed and safety.immediate_intervention_required:
            red_flags = safety.red_flags_detected or []
            flags_text = ", ".join(red_flags) if red_flags else "critical safety concern"
            return (
                f"RED FLAG DETECTED: {flags_text}. "
                f"Safety officer reasoning: {safety.reasoning}".strip()
            )
    return None


def _moderation_as_supervisor_context(
    moderation_output: Any,
    round_number: int,
    *,
    peer_context: PeerContextMode = "nl",
) -> list[AgentRoundOpinion]:
    style = (peer_context or "nl").strip().lower()
    if style == "nl":
        instruction = format_moderation_nl(moderation_output) + "\nRespond to these instructions."
    else:
        moderation_payload = (
            moderation_output.model_dump()
            if hasattr(moderation_output, "model_dump")
            else moderation_output
        )
        instruction = (
            "Oto wnioski i instrukcje od Supervisora z poprzedniej rundy: "
            + json.dumps(moderation_payload, ensure_ascii=False)
            + ". Odpowiedz na nie"
        )

    supervisor_opinion = ClinicalOpinion(
        top_1_diagnosis="maybe",
        evidence_conclusiveness="inconclusive",
        top_3_differential_diagnoses=["yes", "no", "maybe"],
        pros=[instruction],
        cons=[],
        required_further_tests=[],
        confidence_level=0.0,
        sources_used=[],
        red_flags=[],
        missing_information="",
    )
    supervisor_entry = AgentRoundOpinion(
        agent_id="supervisor",
        persona="supervisor",
        round=round_number,
        opinion=supervisor_opinion,
    )
    return [supervisor_entry]
