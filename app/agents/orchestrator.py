"""Debate orchestrator with active supervisor moderation (MAC / MedAgent style rounds)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import json
from typing import Any, Literal

from app.agents.agent import ClinicalAgent
from app.agents.aggregation import opinion_label
from app.agents.models import AgentRoundOpinion, ClinicalOpinion, DebateResult
from app.agents.supervisor_agent import SupervisorAgent

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
EarlyExitFn = Callable[[int, list[AgentRoundOpinion]], bool]

_ARCHITECTURE_BY_MODE: dict[DebateMode, str] = {
    "moderated": "supervised_moderation",
    "peer": "peer_round_robin",
    "hybrid": "hybrid_supervised_peer",
}


class DebateOrchestrator:
    """
    Runs 2-3 rounds of peer debate among ClinicalAgent instances.

    Round 1 ("independent opinion"): every agent answers in parallel with no
    peer context, so nobody anchors on somebody else's first take.

    Round 2+ depends on ``debate_mode``:

    - ``moderated`` (default): supervisor moderates, then all agents revise
      concurrently using supervisor instructions only (legacy behavior).
    - ``peer``: round-robin peer debate; each agent sees prior-round opinions
      plus peers who already spoke this round. No supervisor moderation.
    - ``hybrid``: supervisor moderates, then round-robin peer debate where each
      agent also sees supervisor agreements/contradictions/instructions plus
      peer opinions from the previous round.

    Optional early-exit skips later rounds when the panel already agrees.
    """

    ARCHITECTURE: str = "supervised_moderation"

    def __init__(
        self,
        agents: list[ClinicalAgent],
        *,
        rounds: int = 3,
        debate_mode: DebateMode = "moderated",
        early_exit: EarlyExitFn | None = None,
        agent_concurrency: int = 4,
        supervisor_backend: Any | None = None,
    ) -> None:
        if len(agents) < 2:
            raise ValueError("DebateOrchestrator requires at least 2 agents.")
        if rounds < 2 or rounds > 3:
            raise ValueError("rounds must be 2 or 3.")
        if agent_concurrency < 1:
            raise ValueError("agent_concurrency must be >= 1.")
        if debate_mode not in _ARCHITECTURE_BY_MODE:
            raise ValueError(f"Unsupported debate_mode: {debate_mode}")
        self.agents = agents
        self.rounds = rounds
        self.debate_mode = debate_mode
        self.early_exit = early_exit
        self.agent_concurrency = agent_concurrency
        self.early_exits = 0
        self.ARCHITECTURE = _ARCHITECTURE_BY_MODE[debate_mode]
        self.supervisor = SupervisorAgent(
            backend=supervisor_backend if supervisor_backend is not None else agents[0].backend
        )

    async def run(self, patient_case: str) -> DebateResult:
        history: list[list[AgentRoundOpinion]] = []
        supervisor_moderation: list = []
        pending_red_flag_instruction: str | None = None

        for round_number in range(1, self.rounds + 1):
            if round_number == 1:
                round_opinions = await self._run_independent_round(patient_case)
            else:
                previous_round = history[-1]
                moderation = None
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

            if (
                round_number < self.rounds
                and self.early_exit is not None
                and self.early_exit(round_number, round_opinions)
            ):
                self.early_exits += 1
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
        opinion = await agent.generate_opinion(patient_case, context=context)
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
