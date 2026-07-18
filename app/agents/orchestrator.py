"""Supervisor-free debate orchestrator (MAC / MedAgent style rounds)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from app.agents.agent import ClinicalAgent
from app.agents.aggregation import opinion_label
from app.agents.models import AgentRoundOpinion, ClinicalOpinion, DebateResult


DEFAULT_PERSONAS: tuple[tuple[str, str], ...] = (
    ("generalist", "generalist"),
    ("evidence_skeptic", "evidence_skeptic"),
    ("differential_expander", "differential_expander"),
    ("safety_officer", "safety_officer"),
)

EarlyExitFn = Callable[[int, list[AgentRoundOpinion]], bool]


class DebateOrchestrator:
    """
    Runs 2–3 rounds of peer debate among ClinicalAgent instances.

    No supervisor: after the final round, returns full history and each agent's
    last opinion. Optional early-exit skips later rounds when the panel already agrees.
    """

    def __init__(
        self,
        agents: list[ClinicalAgent],
        *,
        rounds: int = 3,
        early_exit: EarlyExitFn | None = None,
        agent_concurrency: int = 4,
    ) -> None:
        if len(agents) < 2:
            raise ValueError("DebateOrchestrator requires at least 2 agents.")
        if rounds < 2 or rounds > 3:
            raise ValueError("rounds must be 2 or 3.")
        if agent_concurrency < 1:
            raise ValueError("agent_concurrency must be >= 1.")
        self.agents = agents
        self.rounds = rounds
        self.early_exit = early_exit
        self.agent_concurrency = agent_concurrency
        self.early_exits = 0

    async def run(self, patient_case: str) -> DebateResult:
        history: list[list[AgentRoundOpinion]] = []
        previous: list[AgentRoundOpinion] | None = None

        for round_number in range(1, self.rounds + 1):
            round_opinions = await self._run_round(
                patient_case=patient_case,
                round_number=round_number,
                previous=previous,
            )
            history.append(round_opinions)
            previous = round_opinions
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
        )

    async def _run_round(
        self,
        *,
        patient_case: str,
        round_number: int,
        previous: list[AgentRoundOpinion] | None,
    ) -> list[AgentRoundOpinion]:
        semaphore = asyncio.Semaphore(self.agent_concurrency)

        async def _one(agent: ClinicalAgent) -> AgentRoundOpinion:
            async with semaphore:
                context = _peer_context(agent.agent_id, previous)
                opinion = await agent.generate_opinion(patient_case, context=context)
                return AgentRoundOpinion(
                    agent_id=agent.agent_id,
                    persona=agent.persona,
                    round=round_number,
                    opinion=opinion,
                )

        return list(await asyncio.gather(*[_one(agent) for agent in self.agents]))


def labels_unanimous(round_opinions: list[AgentRoundOpinion]) -> bool:
    labels = [opinion_label(entry.opinion) for entry in round_opinions]
    labels = [label for label in labels if label is not None]
    return bool(labels) and len(set(labels)) == 1


def _peer_context(
    agent_id: str,
    previous: list[AgentRoundOpinion] | None,
) -> list[ClinicalOpinion] | None:
    if not previous:
        return None
    return [entry.opinion for entry in previous if entry.agent_id != agent_id]
