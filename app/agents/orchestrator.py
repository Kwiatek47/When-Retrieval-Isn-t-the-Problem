"""Debate orchestrator with active supervisor moderation (MAC / MedAgent style rounds)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import json
from typing import Any, Literal

from app.agents.agent import ClinicalAgent
from app.agents.aggregation import opinion_label
from app.agents.models import AgentRoundOpinion, ClinicalOpinion, DebateResult, SupervisorModerationOutput
from app.agents.supervisor_agent import SupervisorAgent
from app.agents.uncertainty import _label_entropy

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
EarlyExitFn = Callable[[int, list[AgentRoundOpinion], str], bool]

# Blind-critic personas: no BioLinkBERT hint in round 1.
_BLIND_HINT_PERSONAS_R1: frozenset[str] = frozenset({"uncertainty_advocate"})

_ARCHITECTURE_BY_MODE: dict[DebateMode, str] = {
    "moderated": "supervised_moderation",
    "peer": "peer_round_robin",
    "hybrid": "hybrid_supervised_peer",
}


class DebateOrchestrator:
    """
    Runs multi-round peer debate among ClinicalAgent instances.

    Round 1 ("independent opinion"): every agent answers in parallel with no
    peer context, so nobody anchors on somebody else's first take. In PubMedQA
    mode, ``uncertainty_advocate`` is a round-1 "blind critic": BioLinkBERT
    hints are suppressed for that agent only (restored from round 2).

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
        self.early_exit = early_exit
        self.agent_concurrency = agent_concurrency
        self.early_exits = 0
        self.adaptive_stops = 0
        self.ARCHITECTURE = _ARCHITECTURE_BY_MODE[debate_mode]
        self.supervisor = SupervisorAgent(
            backend=supervisor_backend if supervisor_backend is not None else agents[0].backend
        )

    async def run(self, patient_case: str) -> DebateResult:
        history: list[list[AgentRoundOpinion]] = []
        supervisor_moderation: list[SupervisorModerationOutput] = []
        pending_red_flag_instruction: str | None = None

        for round_number in range(1, self.max_rounds + 1):
            if round_number == 1:
                round_opinions = await self._run_independent_round(patient_case)
            else:
                previous_round = history[-1]
                supervisor_prefix: list[AgentRoundOpinion] = []

                if self.debate_mode in ("moderated", "hybrid"):
                    agents_opinions = {
                        entry.agent_id: entry.opinion.model_dump() for entry in previous_round
                    }
                    moderation = await self.supervisor.moderate_round(
                        patient_case=patient_case,
                        agents_opinions=agents_opinions,
                    )
                    if pending_red_flag_instruction:
                        moderation.round_instructions.insert(0, pending_red_flag_instruction)
                    supervisor_moderation.append(moderation)
                    supervisor_prefix = _moderation_as_supervisor_context(
                        moderation_output=moderation,
                        round_number=round_number,
                    )

                if self.debate_mode == "moderated":
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
                        context_prefix=supervisor_prefix if self.debate_mode == "hybrid" else None,
                    )
            history.append(round_opinions)

            pending_red_flag_instruction = _pending_red_flag_instruction(round_opinions)

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

        return DebateResult(
            patient_case=patient_case,
            rounds=history,
            final_opinions=history[-1],
            supervisor_moderation=supervisor_moderation,
        )

    async def _run_independent_round(self, patient_case: str) -> list[AgentRoundOpinion]:
        """Round 1: every agent answers concurrently with no peer context."""
        semaphore = asyncio.Semaphore(self.agent_concurrency)

        async def _one(agent: ClinicalAgent) -> AgentRoundOpinion:
            async with semaphore:
                return await self._speak(agent, patient_case, round_number=1, context=None)

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

    async def _speak(
        self,
        agent: ClinicalAgent,
        patient_case: str,
        *,
        round_number: int,
        context: list[AgentRoundOpinion] | None,
    ) -> AgentRoundOpinion:
        # Round-1 "blind critic": hide BioLinkBERT from uncertainty_advocate so
        # they judge the abstract without classifier pressure.
        include_hint = not (
            round_number == 1
            and (
                agent.agent_id in _BLIND_HINT_PERSONAS_R1
                or agent.persona in _BLIND_HINT_PERSONAS_R1
            )
        )
        opinion = await agent.generate_opinion(
            patient_case,
            context=context,
            include_evidence_hint=include_hint,
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


DEFAULT_ADVOCATE_VETO_CONFIDENCE = 0.65

# Light heuristic: abstract language that often signals inconclusiveness.
INCONCLUSIVE_ABSTRACT_PHRASES: tuple[str, ...] = (
    "small sample size",
    "further research is needed",
    "no statistically significant difference",
    "limitation",
    "preliminary",
)


def _safe_confidence(value: Any, default: float = 1.0) -> float:
    """Parse confidence to float in [0, 1]; fall back to *default* on bad input."""
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return default
    if conf != conf:  # NaN
        return default
    return min(1.0, max(0.0, conf))


def abstract_suggests_inconclusive(
    patient_case: str,
    *,
    phrases: tuple[str, ...] = INCONCLUSIVE_ABSTRACT_PHRASES,
) -> bool:
    """Return True if the case/abstract contains inconclusiveness cue phrases."""
    text = (patient_case or "").lower()
    if not text:
        return False
    return any(phrase in text for phrase in phrases)


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
    (e.g. ``further research is needed``).

    BioLinkBERT is intentionally ignored (option A).
    """
    if not round_opinions:
        return False

    if patient_case and abstract_suggests_inconclusive(patient_case):
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
    - supervisor reported unresolved contradictions.
    """
    if moderation is not None and moderation.contradictions:
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
) -> list[AgentRoundOpinion]:
    moderation_payload = (
        moderation_output.model_dump()
        if hasattr(moderation_output, "model_dump")
        else moderation_output
    )
    moderation_json = json.dumps(moderation_payload, ensure_ascii=False)

    instruction = (
        "Oto wnioski i instrukcje od Supervisora z poprzedniej rundy: "
        + moderation_json
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
