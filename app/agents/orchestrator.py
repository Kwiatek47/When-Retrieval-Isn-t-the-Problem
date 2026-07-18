"""Supervisor-free debate orchestrator (MAC / MedAgent style rounds)."""

from __future__ import annotations

import asyncio

from app.agents.agent import ClinicalAgent
from app.agents.models import AgentRoundOpinion, ClinicalOpinion, DebateResult


DEFAULT_PERSONAS: tuple[tuple[str, str], ...] = (
    ("generalist", "generalist"),
    ("evidence_skeptic", "evidence_skeptic"),
    ("differential_expander", "differential_expander"),
    ("safety_officer", "safety_officer"),
)


class DebateOrchestrator:
    """
    Runs 2–3 rounds of peer debate among ClinicalAgent instances.

    No supervisor: after the final round, returns full history and each agent's
    last opinion.
    """

    def __init__(
        self,
        agents: list[ClinicalAgent],
        *,
        rounds: int = 3,
    ) -> None:
        if len(agents) < 2:
            raise ValueError("DebateOrchestrator requires at least 2 agents.")
        if rounds < 2 or rounds > 3:
            raise ValueError("rounds must be 2 or 3.")
        self.agents = agents
        self.rounds = rounds

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
        async def _one(agent: ClinicalAgent) -> AgentRoundOpinion:
            context = _peer_context(agent.agent_id, previous)
            opinion = await agent.generate_opinion(patient_case, context=context)
            return AgentRoundOpinion(
                agent_id=agent.agent_id,
                persona=agent.persona,
                round=round_number,
                opinion=opinion,
            )

        # Parallel within each round (including round 1 independent opinions).
        return list(await asyncio.gather(*[_one(agent) for agent in self.agents]))


def _peer_context(
    agent_id: str,
    previous: list[AgentRoundOpinion] | None,
) -> list[ClinicalOpinion] | None:
    if not previous:
        return None
    return [entry.opinion for entry in previous if entry.agent_id != agent_id]
